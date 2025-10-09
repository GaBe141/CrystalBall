"""
Consensus Metrics - Performance metrics for consensus systems
"""

from typing import Any

import numpy as np
from dataclasses import dataclass

from ..logutil import get_logger

logger = get_logger("crystalball.consensus.metrics")


@dataclass
class ConsensusMetrics:
    """Container for consensus performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_score: float
    diversity_score: float
    stability_score: float
    
    def overall_score(self) -> float:
        """Compute overall performance score."""
        return (
            0.3 * self.accuracy +
            0.2 * self.precision +
            0.2 * self.recall +
            0.1 * self.confidence_score +
            0.1 * self.diversity_score +
            0.1 * self.stability_score
        )


def compute_consensus_metrics(
    consensus_history: list[dict[str, Any]], actual_results: dict[str, Any] | None = None
) -> ConsensusMetrics:
    """Compute comprehensive metrics for consensus performance."""
    
    if not consensus_history:
        return ConsensusMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Extract scores from history
    confidences = []
    variances = []
    top_models = []
    
    for record in consensus_history:
        if record.get('consensus_scores'):
            scores = record['consensus_scores']
            if scores:
                confidences.append(scores[0]['confidence'])
                variances.append(scores[0]['variance'])
                top_models.append(scores[0]['model'])
    
    # Compute basic metrics
    confidence_score = np.mean(confidences) if confidences else 0.0
    diversity_score = len(set(top_models)) / len(top_models) if top_models else 0.0
    stability_score = 1.0 - np.std(variances) if variances else 0.0
    
    # If actual results provided, compute accuracy metrics
    if actual_results:
        accuracy, precision, recall, f1 = compute_accuracy_metrics(
            consensus_history, actual_results
        )
    else:
        accuracy = precision = recall = f1 = 0.0
    
    return ConsensusMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confidence_score=confidence_score,
        diversity_score=diversity_score,
        stability_score=stability_score
    )


def compute_accuracy_metrics(consensus_history, actual_results):
    """Compute accuracy, precision, recall, and F1 score."""
    
    predictions = []
    actuals = []
    
    for record in consensus_history:
        series_name = record.get('series_name')
        if series_name in actual_results and record.get('consensus_scores'):
            predicted = record['consensus_scores'][0]['model']
            actual = actual_results[series_name]
            predictions.append(predicted)
            actuals.append(actual)
    
    if not predictions:
        return 0.0, 0.0, 0.0, 0.0
    
    # Binary classification metrics (correct vs incorrect)
    correct = sum(p == a for p, a in zip(predictions, actuals))
    total = len(predictions)
    
    accuracy = correct / total
    
    # For multiclass, compute macro-averaged precision/recall
    unique_models = list(set(predictions + actuals))
    precisions = []
    recalls = []
    
    for model in unique_models:
        tp = sum(p == model and a == model for p, a in zip(predictions, actuals))
        fp = sum(p == model and a != model for p, a in zip(predictions, actuals))
        fn = sum(p != model and a == model for p, a in zip(predictions, actuals))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return accuracy, avg_precision, avg_recall, f1


def analyze_consensus_trends(consensus_history, window_size=10):
    """Analyze trends in consensus performance over time."""
    
    if len(consensus_history) < window_size:
        return {"error": "Insufficient history for trend analysis"}
    
    # Extract time series of key metrics
    timestamps = []
    confidences = []
    variances = []
    model_counts = []
    
    for record in consensus_history:
        if record.get('consensus_scores') and record.get('timestamp'):
            timestamps.append(record['timestamp'])
            scores = record['consensus_scores']
            
            avg_confidence = np.mean([s['confidence'] for s in scores])
            avg_variance = np.mean([s['variance'] for s in scores])
            
            confidences.append(avg_confidence)
            variances.append(avg_variance)
            model_counts.append(len(scores))
    
    # Compute rolling statistics
    def rolling_mean(data, window):
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    confidence_trend = rolling_mean(confidences, window_size)
    variance_trend = rolling_mean(variances, window_size)
    count_trend = rolling_mean(model_counts, window_size)
    
    # Detect trends (positive, negative, stable)
    def detect_trend(values):
        if len(values) < 2:
            return "stable"
        
        recent = values[-window_size//2:]
        early = values[:window_size//2]
        
        if np.mean(recent) > np.mean(early) * 1.05:
            return "increasing"
        elif np.mean(recent) < np.mean(early) * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    return {
        "confidence_trend": detect_trend(confidences),
        "variance_trend": detect_trend(variances),
        "model_count_trend": detect_trend(model_counts),
        "latest_metrics": {
            "avg_confidence": confidences[-1] if confidences else 0.0,
            "avg_variance": variances[-1] if variances else 0.0,
            "model_count": model_counts[-1] if model_counts else 0
        },
        "trend_analysis_window": window_size,
        "data_points": len(consensus_history)
    }