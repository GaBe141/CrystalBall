from __future__ import annotations

import glob
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import load_config
from .logutil import get_logger

LOGGER = get_logger("crystalball.llm_consensus")


PROMPT_TEMPLATE = """You are a model evaluation panelist. Given a table of forecasting models and their metrics for a CPI-like time series, produce a JSON verdict ranking models by expected future predictive accuracy.

Consider:
- Out-of-sample error (lower is better): RMSE, MAE, MAPE (if present)
- Cross-validation metrics (cv_rmse, cv_mae) if provided
- Robustness and stability (penalize models with many errors/NaNs)
- Prefer simpler models when metrics are close (Occam’s razor)

Input:
Series: {series_name}
Metrics table (CSV):
{table_csv}

Respond ONLY with JSON of this shape:
{{
  "panelist": "{panel_name}",
  "scores": [
    {{"model": "arima", "score": 0.82, "rank": 1, "notes": "lowest rmse"}},
    {{"model": "ets", "score": 0.78, "rank": 2, "notes": "close second"}}
  ]
}}
Where score is 0..1 (higher is better). If uncertain, explain briefly in notes.
"""


class BaseProvider:
    name: str = "base"

    def available(self) -> bool:
        return True

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        raise NotImplementedError


class HeuristicProvider(BaseProvider):
    name = "heuristic"

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        # Composite score from available error metrics; lower error => higher score
        metrics_pref_low = ["rmse", "mae", "mape", "cv_rmse", "cv_mae"]
        work = df.copy()
        if "model" not in work.columns:
            return {"panelist": self.name, "scores": [], "error": "no_model_column"}
        for c in metrics_pref_low:
            if c in work.columns:
                col = pd.to_numeric(work[c], errors="coerce")
                if col.notna().sum() == 0:
                    continue
                q1, q9 = col.quantile(0.1), col.quantile(0.9)
                rng = float(q9 - q1) if pd.notna(q1) and pd.notna(q9) else 0.0
                if rng <= 0:
                    rng = max(float(col.max() - col.min()), 1e-9)
                norm = (q9 - col) / max(rng, 1e-9)
                work[f"score_{c}"] = norm.clip(0.0, 1.0)
        score_cols = [c for c in work.columns if c.startswith("score_")]
        if not score_cols:
            scores = [{"model": str(m), "score": 0.5, "rank": None, "notes": "no numeric metrics"} for m in work["model"]]
        else:
            work["score"] = work[score_cols].mean(axis=1, skipna=True).fillna(0.0).clip(0.0, 1.0)
            work = work.sort_values("score", ascending=False).reset_index(drop=True)
            scores = []
            for i, row in work.iterrows():
                notes_parts = []
                for c in ["rmse", "mae", "mape"]:
                    if c in work.columns and pd.notna(row.get(c, None)):
                        try:
                            notes_parts.append(f"{c}={float(row[c]):.4g}")
                        except Exception:
                            pass
                scores.append({
                    "model": str(row["model"]),
                    "score": float(row["score"]),
                    "rank": int(i + 1),
                    "notes": ", ".join(notes_parts)
                })
        return {"panelist": self.name, "scores": scores}


class OpenAIProvider(BaseProvider):
    name = "openai"

    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "OPENAI_API_KEY not set"}
        try:
            from openai import OpenAI
            client = OpenAI()
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert forecast evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "ANTHROPIC_API_KEY not set"}
        try:
            import anthropic
            client = anthropic.Anthropic()
            model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            msg = client.messages.create(
                model=model,
                max_tokens=800,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


class GoogleProvider(BaseProvider):
    name = "google"

    def available(self) -> bool:
        return bool(os.environ.get("GOOGLE_API_KEY"))

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "GOOGLE_API_KEY not set"}
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            model_name = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")
            model = genai.GenerativeModel(model_name)
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            resp = model.generate_content(prompt)
            text = resp.text.strip() if hasattr(resp, "text") else str(resp)
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


class AzureOpenAIProvider(BaseProvider):
    name = "azure"

    def available(self) -> bool:
        return all(
            os.environ.get(k)
            for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
        )

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "Azure OpenAI env not set"}
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
            )
            deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are an expert forecast evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


class MistralProvider(BaseProvider):
    name = "mistral"

    def available(self) -> bool:
        return bool(os.environ.get("MISTRAL_API_KEY"))

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "MISTRAL_API_KEY not set"}
        try:
            from mistralai import Mistral
            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
            model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            resp = client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
            text = resp.choices[0].message.content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


class CohereProvider(BaseProvider):
    name = "cohere"

    def available(self) -> bool:
        return bool(os.environ.get("COHERE_API_KEY"))

    def score(self, series_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.available():
            return {"panelist": self.name, "scores": [], "error": "COHERE_API_KEY not set"}
        try:
            import cohere
            co = cohere.Client(os.environ["COHERE_API_KEY"])
            model = os.environ.get("COHERE_MODEL", "command-r-plus")
            table_csv = df.to_csv(index=False)
            prompt = PROMPT_TEMPLATE.format(series_name=series_name, table_csv=table_csv, panel_name=self.name)
            resp = co.chat(model=model, message=prompt, temperature=0.2)
            text = getattr(resp, "text", None) or getattr(resp, "message", "")
            text = str(text).strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                return {"panelist": self.name, "scores": [], "error": "no_json_in_response", "raw": text[:1000]}
            parsed = json.loads(text[start:end+1])
            return _normalize_panel(self.name, parsed)
        except Exception as e:
            return {"panelist": self.name, "scores": [], "error": repr(e)}


def _normalize_panel(panel_name: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    scores = []
    for s in parsed.get("scores", []):
        try:
            scores.append({
                "model": str(s["model"]),
                "score": float(s["score"]),
                "rank": int(s.get("rank") or 0) or None,
                "notes": str(s.get("notes", ""))[:400],
            })
        except Exception:
            continue
    return {"panelist": panel_name, "scores": scores}


def _load_rankings_tables(processed_dir: str) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
    files = sorted(glob.glob(os.path.join(processed_dir, "*_rankings*.csv")))
    out: List[Tuple[str, pd.DataFrame, pathlib.Path]] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "model" not in df.columns:
                continue
            series_name = pathlib.Path(f).stem.replace("_rankings", "")
            out.append((series_name, df, pathlib.Path(f)))
        except Exception:
            continue
    return out


def _aggregate_consensus(panel_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pr in panel_results:
        panel = pr.get("panelist", "unknown")
        for s in pr.get("scores", []):
            rows.append({"panelist": panel, "model": s["model"], "score": float(s["score"])})
    if not rows:
        return pd.DataFrame(columns=["model", "consensus_score", "consensus_rank", "votes", "score_variance", "confidence"])
    df = pd.DataFrame(rows)
    grouped = df.groupby("model").agg(
        consensus_score=("score", "mean"),
        votes=("score", "count"),
        score_variance=("score", "var"),
    ).reset_index()
    # Confidence heuristic: more votes and lower variance => higher confidence in [0,1]
    if not grouped.empty:
        # Normalize votes and variance to 0..1
        v_min, v_max = grouped["votes"].min(), grouped["votes"].max()
        grouped["votes_norm"] = (grouped["votes"] - v_min) / max(v_max - v_min, 1e-9)
        var = grouped["score_variance"].fillna(0.0)
        var_max = float(var.max()) if len(var) else 1.0
        var_norm = (var_max - var) / max(var_max, 1e-9)
        grouped["confidence"] = 0.6 * grouped["votes_norm"] + 0.4 * var_norm
    else:
        grouped["confidence"] = []
    agg = grouped.sort_values("consensus_score", ascending=False).copy()
    agg["consensus_rank"] = range(1, len(agg) + 1)
    return agg[["model", "consensus_score", "consensus_rank", "votes", "score_variance", "confidence"]]


def run_llm_consensus(providers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run LLM consensus over all *_rankings*.csv and write results under exports/llm_consensus.

    providers: optional list to select among [heuristic, openai, anthropic, google, azure, mistral, cohere]
    """
    cfg = load_config()
    processed = str(cfg.paths.processed_dir)
    out_dir = os.path.join(str(cfg.paths.exports_dir), "llm_consensus")
    os.makedirs(out_dir, exist_ok=True)

    prov_map: Dict[str, BaseProvider] = {
        "heuristic": HeuristicProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "google": GoogleProvider(),
        "azure": AzureOpenAIProvider(),
        "mistral": MistralProvider(),
        "cohere": CohereProvider(),
    }
    chosen = providers or ["heuristic", "openai", "anthropic", "google", "azure", "mistral", "cohere"]
    active = [prov_map[p] for p in chosen if p in prov_map]
    if not active:
        active = [HeuristicProvider()]

    tables = _load_rankings_tables(processed)
    summary_index = []
    for series_name, df, path in tables:
        panel_results: List[Dict[str, Any]] = []
        for prov in active:
            try:
                if not prov.available():
                    panel_results.append({"panelist": prov.name, "scores": [], "error": "unavailable"})
                    continue
                pr = prov.score(series_name, df)
                panel_results.append(pr)
                time.sleep(0.25)
            except Exception as e:
                panel_results.append({"panelist": prov.name, "scores": [], "error": repr(e)})
        consensus = _aggregate_consensus(panel_results)
        base = pathlib.Path(path).stem
        json_path = os.path.join(out_dir, f"{base}_llm_consensus.json")
        csv_path = os.path.join(out_dir, f"{base}_llm_consensus.csv")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "series": series_name,
                    "providers": [p.name for p in active],
                    "panel": panel_results,
                }, f, indent=2)
        except Exception:
            LOGGER.exception("Failed writing JSON consensus for %s", series_name)
        try:
            consensus.to_csv(csv_path, index=False)
        except Exception:
            LOGGER.exception("Failed writing CSV consensus for %s", series_name)
        summary_index.append({"series": series_name, "json": json_path, "csv": csv_path, "providers": [p.name for p in active]})

    overview = os.path.join(out_dir, "consensus_overview.json")
    with open(overview, "w", encoding="utf-8") as f:
        json.dump({"items": summary_index}, f, indent=2)
    return {"out_dir": out_dir, "count": len(summary_index), "overview": overview}
