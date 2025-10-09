"""
Consensus Validator - Validation framework for consensus performance
"""

from dataclasses import dataclass

import numpy as np

from ..logutil import get_logger

logger = get_logger("crystalball.consensus.validator")

# Constants
MIN_MODELS_FOR_CORRELATION = 2


@dataclass
class ValidationResult:
    """Results from consensus validation."""
    accuracy: float
    top_k_accuracy: dict[int, float]
    rank_correlation: float
    confidence_calibration: float
    method_performance: dict[str, float]


class ConsensusValidator:
    """Validates consensus performance using various metrics."""
    
    def __init__(self, engine) -> None:
        self.engine = engine
        self.validation_history = []
    
    def cross_validate_consensus(self, test_data, methods=None) -> ValidationResult:
        """Cross-validate consensus using historical data."""
        methods = methods or ['weighted_average', 'median', 'robust_mean']
        
        method_results = {}
        all_predictions = []
        all_actuals = []
        
        for method in methods:
            method_predictions = []
            method_actuals = []
            
            for test_case in test_data:
                series_name = test_case['series_name']
                rankings_df = test_case['rankings_df']
                actual_best = test_case['actual_best']
                
                try:
                    consensus_scores = self.engine.compute_consensus(
                        series_name, rankings_df, method=method
                    )
                    
                    if consensus_scores:
                        predicted_best = consensus_scores[0].model
                        method_predictions.append(predicted_best)
                        method_actuals.append(actual_best)
                        
                except Exception as e:
                    logger.warning(f"Consensus failed for {series_name}: {e}")
                    continue
            
            if method_predictions:
                accuracy = sum(
                    p == a for p, a in zip(method_predictions, method_actuals, strict=True)
                ) / len(method_predictions)
                method_results[method] = accuracy
                
                if method == methods[0]:
                    all_predictions = method_predictions
                    all_actuals = method_actuals
        
        # Compute comprehensive metrics
        return self._compute_validation_metrics(
            all_predictions, all_actuals, method_results, test_data
        )
    
    def _compute_validation_metrics(self, predictions, actuals, method_results, test_data):
        """Compute comprehensive validation metrics."""
        
        # Basic accuracy
        accuracy = (
            sum(p == a for p, a in zip(predictions, actuals, strict=True)) / len(predictions) 
            if predictions else 0.0
        )
        
        # Top-k accuracy
        top_k_accuracy = self._compute_top_k_accuracy(test_data)
        
        # Rank correlation
        rank_correlation = self._compute_rank_correlation(test_data)
        
        # Confidence calibration
        confidence_calibration = self._compute_confidence_calibration(test_data)
        
        result = ValidationResult(
            accuracy=accuracy,
            top_k_accuracy=top_k_accuracy,
            rank_correlation=rank_correlation,
            confidence_calibration=confidence_calibration,
            method_performance=method_results
        )
        
        self.validation_history.append(result)
        return result
    
    def _compute_top_k_accuracy(self, test_data):
        """Compute top-k accuracy for different values of k."""
        top_k_results = {}
        
        for k in [1, 3, 5]:
            correct = 0
            total = 0
            
            for test_case in test_data:
                try:
                    consensus_scores = self.engine.compute_consensus(
                        test_case['series_name'], 
                        test_case['rankings_df'],
                        method='weighted_average'
                    )
                    
                    if len(consensus_scores) >= k:
                        top_k_models = [cs.model for cs in consensus_scores[:k]]
                        if test_case['actual_best'] in top_k_models:
                            correct += 1
                        total += 1
                        
                except Exception:
                    continue
            
            top_k_results[k] = correct / total if total > 0 else 0.0
            
        return top_k_results
    
    def _compute_rank_correlation(self, test_data):
        """Compute rank correlation between consensus and actual rankings."""
        correlations = []
        
        for test_case in test_data:
            if 'actual_rankings' not in test_case:
                continue
                
            try:
                consensus_scores = self.engine.compute_consensus(
                    test_case['series_name'], 
                    test_case['rankings_df'],
                    method='weighted_average'
                )
                
                consensus_ranks = {cs.model: cs.rank for cs in consensus_scores}
                actual_ranks = test_case['actual_rankings']
                
                # Find common models
                common_models = set(consensus_ranks.keys()) & set(actual_ranks.keys())
                
                if len(common_models) > MIN_MODELS_FOR_CORRELATION:
                    consensus_vals = [consensus_ranks[m] for m in common_models]
                    actual_vals = [actual_ranks[m] for m in common_models]
                    
                    correlation = np.corrcoef(consensus_vals, actual_vals)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
                        
            except Exception:
                continue
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _compute_confidence_calibration(self, test_data):
        """Compute how well consensus confidence matches actual accuracy."""
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            bin_low, bin_high = confidence_bins[i], confidence_bins[i + 1]
            bin_predictions = []
            bin_actuals = []
            bin_conf_values = []
            
            for test_case in test_data:
                try:
                    consensus_scores = self.engine.compute_consensus(
                        test_case['series_name'], 
                        test_case['rankings_df'],
                        method='weighted_average'
                    )
                    
                    if consensus_scores:
                        top_score = consensus_scores[0]
                        if bin_low <= top_score.confidence < bin_high:
                            bin_predictions.append(top_score.model)
                            bin_actuals.append(test_case['actual_best'])
                            bin_conf_values.append(top_score.confidence)
                            
                except Exception:
                    continue
            
            if bin_predictions:
                bin_accuracy = (
                    sum(p == a for p, a in zip(bin_predictions, bin_actuals, strict=True)) 
                    / len(bin_predictions)
                )
                bin_confidence = np.mean(bin_conf_values)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        # Compute Expected Calibration Error (ECE)
        if bin_accuracies and bin_confidences:
            ece = np.mean([
                abs(acc - conf) 
                for acc, conf in zip(bin_accuracies, bin_confidences, strict=True)
            ])
            return 1.0 - ece
        
        return 0.0
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_history:
            return "No validation results available."
        
        latest = self.validation_history[-1]
        
        report = f"""
# Consensus Validation Report

## Overall Performance
- **Accuracy**: {latest.accuracy:.3f}
- **Top-1 Accuracy**: {latest.top_k_accuracy.get(1, 0):.3f}
- **Top-3 Accuracy**: {latest.top_k_accuracy.get(3, 0):.3f}
- **Top-5 Accuracy**: {latest.top_k_accuracy.get(5, 0):.3f}

## Advanced Metrics
- **Rank Correlation**: {latest.rank_correlation:.3f}
- **Confidence Calibration**: {latest.confidence_calibration:.3f}

## Method Comparison
"""
        
        for method, performance in latest.method_performance.items():
            report += f"- **{method.replace('_', ' ').title()}**: {performance:.3f}\n"
        
        if len(self.validation_history) > 1:
            recent_accuracy = [r.accuracy for r in self.validation_history[-5:]]
            report += (
                f"\n## Recent Trend\n- **Average Accuracy (last 5)**: "
                f"{np.mean(recent_accuracy):.3f}\n"
            )
        
        return report