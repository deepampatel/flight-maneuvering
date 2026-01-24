"""
ML module for AI-based threat assessment and guidance policies.

This module provides:
- ThreatModel: ONNX-based threat scoring
- GuidanceModel: RL-based guidance policy
- Feature extraction utilities
"""

from .inference import ThreatModel, GuidanceModel, MLConfig, get_model_registry
from .features import (
    extract_threat_features,
    extract_guidance_features,
    ThreatFeatures,
    GuidanceFeatures,
)

__all__ = [
    "ThreatModel",
    "GuidanceModel",
    "MLConfig",
    "get_model_registry",
    "extract_threat_features",
    "extract_guidance_features",
    "ThreatFeatures",
    "GuidanceFeatures",
]
