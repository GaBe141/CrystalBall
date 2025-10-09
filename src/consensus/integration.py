"""
Enhanced Consensus Integration - Integrates with existing CrystalBall pipeline
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_config
from ..logutil import get_logger
from .engine import ConsensusEngine
from .validator import ConsensusValidator
from ..llm_consensus import (
    HeuristicProvider, OpenAIProvider, AnthropicProvider, 
    GoogleProvider, AzureOpenAIProvider, MistralProvider, CohereProvider
)

logger = get_logger("crystalball.consensus.integration")


class EnhancedConsensusRunner:
    """Enhanced consensus runner that integrates with existing CrystalBall pipeline."""
    
    def __init__(
        self, providers: list[Any] | None = None, methods: list[str] | None = None
    ) -> None:
        """Initialize enhanced consensus runner."""
        self.methods = methods or ["weighted_average", "median", "robust_mean"]
        
        # Initialize all available providers
        if providers is None:
            self.providers = [
                HeuristicProvider(),
                OpenAIProvider(),
                AnthropicProvider(),
                GoogleProvider(),
                AzureOpenAIProvider(),
                MistralProvider(),
                CohereProvider()
            ]
        else:
            self.providers = providers
            
        # Initialize engines for each method
        self.engines = {
            method: ConsensusEngine(self.providers) for method in self.methods
        }
        
        # Initialize validator
        self.validator = ConsensusValidator(self.engines[self.methods[0]])
        
        # Load config
        self.config = load_config()
        
    def run_enhanced_consensus(
        self,
        rankings_dir: str | None = None,
        output_dir: str | None = None
    ) -> dict[str, Any]:
        """Run enhanced consensus on all rankings files."""
        if rankings_dir is None:
            rankings_dir = self.config.paths.processed_dir
        
        if output_dir is None:
            output_dir = os.path.join(self.config.paths.exports_dir, "enhanced_consensus")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all rankings files
        rankings_files = list(Path(rankings_dir).glob("*_rankings*.csv"))
        logger.info(f"Found {len(rankings_files)} rankings files")
        
        if not rankings_files:
            logger.warning("No rankings files found")
            return {"error": "No rankings files found"}
        
        results = []
        
        for rankings_file in rankings_files:
            try:
                # Load rankings
                rankings_df = pd.read_csv(rankings_file)
                if "model" not in rankings_df.columns:
                    logger.warning(f"No 'model' column in {rankings_file}")
                    continue
                
                series_name = rankings_file.stem.replace("_rankings", "")
                logger.info(f"Processing {series_name}")
                
                # Run consensus for each method
                method_results = {}
                for method in self.methods:
                    engine = self.engines[method]
                    consensus_scores = engine.compute_consensus(
                        series_name, rankings_df, method=method
                    )
                    method_results[method] = consensus_scores
                
                # Save results
                self._save_enhanced_results(
                    series_name, rankings_df, method_results, output_dir
                )
                
                results.append({
                    "series": series_name,
                    "file": str(rankings_file),
                    "methods": list(method_results.keys()),
                    "consensus_counts": {
                        method: len(scores) for method, scores in method_results.items()
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to process {rankings_file}: {e}")
                continue
        
        # Generate summary
        summary = self._generate_enhanced_summary(results, output_dir)
        
        logger.info(f"Enhanced consensus completed. Results in {output_dir}")
        return summary
    
    def _save_enhanced_results(
        self,
        series_name: str,
        rankings_df: pd.DataFrame,
        method_results: dict[str, Any],
        output_dir: str
    ) -> None:
        """Save enhanced consensus results."""
        
        # Create detailed results for each method
        for method, consensus_scores in method_results.items():
            method_dir = os.path.join(output_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Save CSV with enhanced metrics
            if consensus_scores:
                results_df = pd.DataFrame([
                    {
                        "model": cs.model,
                        "consensus_score": cs.score,
                        "weighted_score": cs.weighted_score,
                        "confidence": cs.confidence,
                        "variance": cs.variance,
                        "provider_count": cs.provider_count,
                        "rank": cs.rank,
                        "notes": cs.notes
                    }
                    for cs in consensus_scores
                ])
                
                csv_path = os.path.join(method_dir, f"{series_name}_enhanced_consensus.csv")
                results_df.to_csv(csv_path, index=False)
                
                # Save detailed JSON
                json_path = os.path.join(method_dir, f"{series_name}_enhanced_consensus.json")
                detailed_results = {
                    "series_name": series_name,
                    "method": method,
                    "consensus_scores": [
                        {
                            "model": cs.model,
                            "score": cs.score,
                            "weighted_score": cs.weighted_score,
                            "confidence": cs.confidence,
                            "variance": cs.variance,
                            "provider_count": cs.provider_count,
                            "rank": cs.rank,
                            "notes": cs.notes
                        }
                        for cs in consensus_scores
                    ],
                    "summary": {
                        "total_models": len(consensus_scores),
                        "avg_confidence": (
                            sum(cs.confidence for cs in consensus_scores) / 
                            len(consensus_scores)
                        ),
                        "avg_variance": (
                            sum(cs.variance for cs in consensus_scores) / 
                            len(consensus_scores)
                        ),
                        "top_model": consensus_scores[0].model if consensus_scores else None
                    }
                }
                
                with open(json_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
        
        # Create comparison across methods
        comparison_results: dict[str, Any] = {
            "series_name": series_name,
            "method_comparison": {}
        }
        
        for method, consensus_scores in method_results.items():
            if consensus_scores:
                comparison_results["method_comparison"][method] = {
                    "top_model": consensus_scores[0].model,
                    "top_score": consensus_scores[0].weighted_score,
                    "confidence": consensus_scores[0].confidence,
                    "total_models": len(consensus_scores)
                }
        
        comparison_path = os.path.join(output_dir, f"{series_name}_method_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
    
    def _generate_enhanced_summary(
        self, results: list[dict[str, Any]], output_dir: str
    ) -> dict[str, Any]:
        """Generate enhanced summary of consensus results."""
        summary = {
            "enhanced_consensus_summary": {
                "total_series": len(results),
                "methods_used": self.methods,
                "providers_available": [p.name for p in self.providers if p.available()],
                "results": results
            }
        }
        
        # Aggregate statistics
        if results:
            method_stats = {}
            for method in self.methods:
                method_counts = [r["consensus_counts"].get(method, 0) for r in results]
                method_stats[method] = {
                    "avg_models_per_series": sum(method_counts) / len(method_counts),
                    "total_consensus_scores": sum(method_counts),
                    "series_processed": len([c for c in method_counts if c > 0])
                }
            
            summary["enhanced_consensus_summary"]["method_statistics"] = method_stats
        
        # Get engine summaries
        engine_summaries = {}
        for method, engine in self.engines.items():
            engine_summaries[method] = engine.get_consensus_summary()
        
        summary["enhanced_consensus_summary"]["engine_summaries"] = engine_summaries
        
        # Save summary
        summary_path = os.path.join(output_dir, "enhanced_consensus_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def validate_consensus_performance(
        self, test_data: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Validate consensus performance using test data."""
        if test_data is None:
            # Generate test data from recent consensus runs
            test_data = self._generate_test_data_from_history()
        
        if not test_data:
            logger.warning("No test data available for validation")
            return {"error": "No test data available"}
        
        validation_result = self.validator.cross_validate_consensus(test_data, self.methods)
        
        # Generate and save report
        report = self.validator.generate_validation_report()
        
        output_dir = os.path.join(self.config.paths.exports_dir, "enhanced_consensus")
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "validation_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save validation results as JSON
        validation_path = os.path.join(output_dir, "validation_results.json")
        with open(validation_path, 'w') as f:
            json.dump({
                "accuracy": validation_result.accuracy,
                "top_k_accuracy": validation_result.top_k_accuracy,
                "rank_correlation": validation_result.rank_correlation,
                "confidence_calibration": validation_result.confidence_calibration,
                "method_performance": validation_result.method_performance
            }, f, indent=2)
        
        logger.info(f"Validation completed. Report saved to {report_path}")
        return {
            "validation_result": validation_result,
            "report_path": report_path,
            "validation_path": validation_path
        }
    
    def _generate_test_data_from_history(self) -> list[dict[str, Any]]:
        """Generate test data from consensus history."""
        test_data = []
        
        # Use history from the first engine
        main_engine = self.engines[self.methods[0]]
        
        for record in main_engine.consensus_history[-10:]:  # Last 10 records
            if record.get('consensus_scores'):
                # Create test case assuming the top consensus model was correct
                top_model = record['consensus_scores'][0]['model']
                
                # Create a mock rankings DataFrame
                rankings_data = []
                for cs in record['consensus_scores']:
                    rankings_data.append({
                        'model': cs['model'],
                        'rmse': 1.0 - cs['score'],  # Inverse score as error
                        'mae': 1.0 - cs['score'],
                    })
                
                rankings_df = pd.DataFrame(rankings_data)
                
                test_data.append({
                    'series_name': record['series_name'],
                    'rankings_df': rankings_df,
                    'actual_best': top_model
                })
        
        return test_data