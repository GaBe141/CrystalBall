"""
Enhanced Consensus Engine - Core implementation for CrystalBall
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..llm_consensus import AnthropicProvider, HeuristicProvider, OpenAIProvider
from ..logutil import get_logger

logger = get_logger("crystalball.consensus.engine")

# Constants
MAX_HISTORY_SIZE = 100
MIN_ROBUST_MEAN_SIZE = 4


@dataclass
class ConsensusScore:
    """Represents a consensus score with uncertainty measures."""
    model: str
    score: float
    confidence: float
    variance: float
    provider_count: int
    weighted_score: float
    rank: int = 0
    notes: str = ""


@dataclass
class ProviderWeight:
    """Dynamic weights for consensus providers."""
    provider: str
    reliability_weight: float
    confidence_weight: float
    specialty_weight: float
    final_weight: float


class ConsensusEngine:
    """Advanced consensus engine with adaptive weighting and validation."""
    
    def __init__(self, providers: list[Any] | None = None) -> None:
        """Initialize consensus engine with providers."""
        self.providers = providers or [
            HeuristicProvider(),
            OpenAIProvider(),
            AnthropicProvider()
        ]
        self.provider_history: dict[str, list[Any]] = {}
        self.consensus_history: list[dict[str, Any]] = []
    
    def compute_consensus(
        self, 
        series_name: str, 
        rankings_df: pd.DataFrame, 
        method: str = "weighted_average", 
        confidence_threshold: float = 0.5
    ) -> list[ConsensusScore]:
        """Compute consensus with multiple aggregation methods."""
        logger.info(f"Computing consensus for {series_name} using {method}")
        
        # Get individual provider scores
        provider_scores = self._collect_provider_scores(series_name, rankings_df)
        
        # Compute dynamic weights
        provider_weights = self._compute_provider_weights(provider_scores, series_name)
        
        # Aggregate scores based on method
        if method == "weighted_average":
            consensus_scores = self._weighted_average_consensus(provider_scores, provider_weights)
        elif method == "median":
            consensus_scores = self._median_consensus(provider_scores)
        elif method == "robust_mean":
            consensus_scores = self._robust_mean_consensus(provider_scores, provider_weights)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
            
        # Filter by confidence and rank
        filtered_scores = [s for s in consensus_scores if s.confidence >= confidence_threshold]
        filtered_scores.sort(key=lambda x: x.weighted_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(filtered_scores):
            score.rank = i + 1
            
        # Store for learning
        self._update_history(series_name, provider_scores, filtered_scores)
        
        return filtered_scores
    
    def _collect_provider_scores(
        self, series_name: str, rankings_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Collect scores from all available providers concurrently."""
        provider_scores = {}
        
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            future_to_provider = {}
            
            for provider in self.providers:
                if provider.available():
                    future = executor.submit(provider.score, series_name, rankings_df)
                    future_to_provider[future] = provider
                    
            for future in as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    result = future.result(timeout=30)
                    provider_scores[provider.name] = result.get('scores', [])
                except Exception as e:
                    logger.warning(f"Provider {provider.name} failed: {e}")
                    provider_scores[provider.name] = []
                    
        return provider_scores
    
    def _compute_provider_weights(
        self, provider_scores: dict[str, Any], series_name: str
    ) -> dict[str, ProviderWeight]:
        """Compute dynamic weights for each provider."""
        weights = {}
        
        for provider_name, scores in provider_scores.items():
            # Base reliability from historical performance
            reliability = self._get_provider_reliability(provider_name)
            
            # Confidence based on score variance
            score_values = [float(s.get('score', 0.5)) for s in scores]
            confidence = 1.0 - (np.var(score_values) if len(score_values) > 1 else 0.0)
            confidence = max(0.0, min(1.0, confidence))
            
            # Specialty weight
            specialty = self._get_specialty_weight(provider_name, series_name)
            
            # Combine weights
            final_weight = float(0.4 * reliability + 0.3 * confidence + 0.3 * specialty)
            
            weights[provider_name] = ProviderWeight(
                provider=provider_name,
                reliability_weight=reliability,
                confidence_weight=float(confidence),
                specialty_weight=specialty,
                final_weight=final_weight
            )
            
        return weights
    
    def _weighted_average_consensus(
        self, provider_scores: dict[str, Any], weights: dict[str, ProviderWeight]
    ) -> list[ConsensusScore]:
        """Compute weighted average consensus."""
        model_scores: dict[str, dict[str, list[Any]]] = {}
        
        # Aggregate scores by model
        for provider_name, scores in provider_scores.items():
            weight = weights.get(provider_name, ProviderWeight("", 1.0, 1.0, 1.0, 1.0)).final_weight
            
            for score_dict in scores:
                model = score_dict.get('model', '')
                score = float(score_dict.get('score', 0.5))
                notes = score_dict.get('notes', '')
                
                if model not in model_scores:
                    model_scores[model] = {'scores': [], 'weights': [], 'notes': []}
                    
                model_scores[model]['scores'].append(score)
                model_scores[model]['weights'].append(weight)
                model_scores[model]['notes'].append(notes)
        
        # Compute consensus for each model
        consensus_scores = []
        for model, data in model_scores.items():
            scores = np.array(data['scores'])
            weights_arr = np.array(data['weights'])
            
            # Weighted average
            weighted_score = np.average(scores, weights=weights_arr) if len(scores) > 0 else 0.5
            
            # Confidence metrics
            variance = np.var(scores) if len(scores) > 1 else 0.0
            confidence = 1.0 / (1.0 + variance)
            
            # Combine notes
            combined_notes = "; ".join([n for n in data['notes'] if n])[:200]
            
            consensus_scores.append(ConsensusScore(
                model=model,
                score=float(np.mean(scores)) if len(scores) > 0 else 0.5,
                confidence=float(confidence),
                variance=float(variance),
                provider_count=len(scores),
                weighted_score=float(weighted_score),
                rank=0,
                notes=combined_notes
            ))
            
        return consensus_scores
    
    def _median_consensus(self, provider_scores: dict[str, Any]) -> list[ConsensusScore]:
        """Compute median-based consensus."""
        model_scores: dict[str, dict[str, list[Any]]] = {}
        
        for _provider_name, scores in provider_scores.items():
            for score_dict in scores:
                model = score_dict.get('model', '')
                score = float(score_dict.get('score', 0.5))
                notes = score_dict.get('notes', '')
                
                if model not in model_scores:
                    model_scores[model] = {'scores': [], 'notes': []}
                    
                model_scores[model]['scores'].append(score)
                model_scores[model]['notes'].append(notes)
        
        consensus_scores = []
        for model, data in model_scores.items():
            scores_arr = np.array(data['scores'])
            
            median_score = float(np.median(scores_arr)) if len(scores_arr) > 0 else 0.5
            variance = float(np.var(scores_arr)) if len(scores_arr) > 1 else 0.0
            confidence = 1.0 / (1.0 + variance)
            
            combined_notes = "; ".join([n for n in data['notes'] if n])[:200]
            
            consensus_scores.append(ConsensusScore(
                model=model,
                score=median_score,
                confidence=confidence,
                variance=variance,
                provider_count=len(data['scores']),
                weighted_score=median_score,
                rank=0,
                notes=combined_notes
            ))
            
        return consensus_scores
    
    def _robust_mean_consensus(
        self, provider_scores: dict[str, Any], weights: dict[str, ProviderWeight]
    ) -> list[ConsensusScore]:
        """Compute robust mean (trimmed mean with weights)."""
        model_scores: dict[str, dict[str, list[Any]]] = {}
        
        for provider_name, scores in provider_scores.items():
            weight = weights.get(provider_name, ProviderWeight("", 1.0, 1.0, 1.0, 1.0)).final_weight
            
            for score_dict in scores:
                model = score_dict.get('model', '')
                score = float(score_dict.get('score', 0.5))
                notes = score_dict.get('notes', '')
                
                if model not in model_scores:
                    model_scores[model] = {'scores': [], 'weights': [], 'notes': []}
                    
                model_scores[model]['scores'].append(score)
                model_scores[model]['weights'].append(weight)
                model_scores[model]['notes'].append(notes)
        
        consensus_scores = []
        for model, data in model_scores.items():
            scores = np.array(data['scores'])
            weights_arr = np.array(data['weights'])
            
            # Trim outliers if enough data points
            if len(scores) > MIN_ROBUST_MEAN_SIZE:
                trim_count = max(1, len(scores) // 10)
                sorted_indices = np.argsort(scores)
                trimmed_indices = sorted_indices[trim_count:-trim_count]
                scores = scores[trimmed_indices]
                weights_arr = weights_arr[trimmed_indices]
            
            weighted_score = np.average(scores, weights=weights_arr) if len(scores) > 0 else 0.5
            variance = float(np.var(scores)) if len(scores) > 1 else 0.0
            confidence = 1.0 / (1.0 + variance)
            
            combined_notes = "; ".join([n for n in data['notes'] if n])[:200]
            
            consensus_scores.append(ConsensusScore(
                model=model,
                score=float(np.mean(scores)) if len(scores) > 0 else 0.5,
                confidence=confidence,
                variance=variance,
                provider_count=len(data['scores']),
                weighted_score=float(weighted_score),
                rank=0,
                notes=combined_notes
            ))
            
        return consensus_scores
    
    def _get_provider_reliability(self, provider_name: str) -> float:
        """Get historical reliability score for provider."""
        history = self.provider_history.get(provider_name, [])
        if not history:
            return 1.0
        return float(np.mean(history))
    
    def _get_specialty_weight(self, provider_name: str, series_name: str) -> float:
        """Get specialty weight based on provider expertise and series type."""
        specialty_map = {
            'heuristic': {'cpi': 0.8, 'financial': 0.9, 'default': 0.7},
            'openai': {'cpi': 0.9, 'financial': 0.8, 'default': 0.9},
            'anthropic': {'cpi': 0.9, 'financial': 0.8, 'default': 0.9},
            'google': {'cpi': 0.8, 'financial': 0.7, 'default': 0.8},
            'azure': {'cpi': 0.9, 'financial': 0.8, 'default': 0.9},
            'mistral': {'cpi': 0.7, 'financial': 0.7, 'default': 0.7},
            'cohere': {'cpi': 0.7, 'financial': 0.7, 'default': 0.7},
        }
        
        provider_specialty = specialty_map.get(provider_name, {'default': 1.0})
        
        # Simple heuristic based on series name
        if 'cpi' in series_name.lower():
            return provider_specialty.get('cpi', provider_specialty['default'])
        elif any(term in series_name.lower() for term in ['stock', 'price', 'financial']):
            return provider_specialty.get('financial', provider_specialty['default'])
        else:
            return provider_specialty['default']
    
    def _update_history(
        self, 
        series_name: str, 
        provider_scores: dict[str, Any], 
        consensus_scores: list[ConsensusScore]
    ) -> None:
        """Update historical tracking for learning."""
        consensus_record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'series_name': series_name,
            'provider_scores': provider_scores,
            'consensus_scores': [
                {
                    'model': cs.model,
                    'score': cs.score,
                    'confidence': cs.confidence,
                    'variance': cs.variance,
                    'weighted_score': cs.weighted_score,
                    'rank': cs.rank,
                    'notes': cs.notes
                }
                for cs in consensus_scores
            ]
        }
        
        self.consensus_history.append(consensus_record)
        
        # Keep only recent history
        if len(self.consensus_history) > MAX_HISTORY_SIZE:
            self.consensus_history = self.consensus_history[-MAX_HISTORY_SIZE:]
    
    def validate_consensus(
        self, actual_best_model: str, consensus_scores: list[ConsensusScore]
    ) -> float:
        """Validate consensus against known best model."""
        if not consensus_scores:
            return 0.0
            
        # Find rank of actual best model in consensus
        for score in consensus_scores:
            if score.model == actual_best_model:
                return float(1.0 / score.rank)
                
        return 0.0
    
    def get_consensus_summary(self) -> dict[str, Any]:
        """Get summary statistics of consensus performance."""
        if not self.consensus_history:
            return {'message': 'No consensus history available'}
            
        recent_records = self.consensus_history[-20:]
        
        if not recent_records:
            return {'message': 'No recent consensus history available'}
        
        try:
            avg_confidence = np.mean([
                np.mean([cs['confidence'] for cs in record['consensus_scores']])
                for record in recent_records
                if record.get('consensus_scores')
            ])
            
            avg_variance = np.mean([
                np.mean([cs['variance'] for cs in record['consensus_scores']])
                for record in recent_records
                if record.get('consensus_scores')
            ])
        except (ValueError, ZeroDivisionError):
            avg_confidence = 0.0
            avg_variance = 0.0
        
        return {
            'total_consensus_runs': len(self.consensus_history),
            'recent_avg_confidence': float(avg_confidence),
            'recent_avg_variance': float(avg_variance),
            'provider_participation': {
                provider: len(self.provider_history.get(provider, []))
                for provider in [
                    'heuristic', 'openai', 'anthropic', 'google', 
                    'azure', 'mistral', 'cohere'
                ]
            }
        }