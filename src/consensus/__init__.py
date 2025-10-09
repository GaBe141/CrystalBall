"""
Enhanced AI Consensus Framework for CrystalBall

This module provides advanced consensus mechanisms building on the existing LLM consensus system.
It includes dynamic weighting, multi-method aggregation, validation, and learning capabilities.
"""

from .engine import ConsensusEngine, ConsensusScore, ProviderWeight
from .integration import EnhancedConsensusRunner
from .metrics import ConsensusMetrics
from .validator import ConsensusValidator, ValidationResult

__all__ = [
    "ConsensusEngine",
    "ConsensusScore", 
    "ProviderWeight",
    "ConsensusValidator",
    "ValidationResult",
    "EnhancedConsensusRunner",
    "ConsensusMetrics",
]