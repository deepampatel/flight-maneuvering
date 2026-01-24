"""
ONNX Model Inference for ML-based Threat Assessment and Guidance

This module provides:
- ThreatModel: Neural network for threat scoring
- GuidanceModel: RL policy for guidance commands
- Model loading/unloading management

ONNX Runtime provides cross-platform inference with:
- CPU execution (default)
- GPU acceleration (optional)
- Optimized graph execution
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import numpy as np

from ..vector import Vec3
from .features import (
    ThreatFeatures,
    GuidanceFeatures,
    extract_threat_features,
    extract_batch_threat_features,
    extract_guidance_features,
)

logger = logging.getLogger(__name__)

# Try to import onnxruntime, but don't fail if not installed
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    logger.warning("onnxruntime not installed - ML features will use fallback mode")


@dataclass
class MLConfig:
    """Configuration for ML model loading and inference."""
    model_path: str = ""
    model_type: str = "threat_assessment"  # "threat_assessment" or "guidance"
    device: str = "cpu"  # "cpu" or "cuda"
    num_threads: int = 4
    enable_profiling: bool = False


@dataclass
class ThreatPrediction:
    """Output from threat assessment model."""
    target_id: str
    threat_score: float        # 0-100
    confidence: float          # 0-1 model confidence
    threat_level: str          # "critical", "high", "medium", "low"
    feature_importances: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "threat_score": round(self.threat_score, 1),
            "confidence": round(self.confidence, 3),
            "threat_level": self.threat_level,
            "feature_importances": self.feature_importances,
        }


@dataclass
class GuidancePrediction:
    """Output from guidance policy model."""
    acceleration: Vec3         # Commanded acceleration
    confidence: float          # 0-1 policy confidence
    action_probabilities: Optional[np.ndarray] = None  # For discrete action spaces

    def to_dict(self) -> dict:
        return {
            "acceleration": self.acceleration.to_dict(),
            "confidence": round(self.confidence, 3),
        }


class ThreatModel:
    """
    ONNX-based threat assessment model.

    Input: ThreatFeatures (18 features)
    Output: threat_score (0-100), confidence (0-1)

    If no model is loaded, falls back to rule-based scoring.
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.session: Optional[Any] = None
        self.input_name: str = "input"
        self.output_names: List[str] = ["threat_score", "confidence"]
        self.model_loaded: bool = False
        self.model_path: str = ""

    def load(self, path: str) -> bool:
        """
        Load ONNX model from file.

        Args:
            path: Path to .onnx file

        Returns:
            True if loaded successfully
        """
        if not ONNX_AVAILABLE:
            logger.warning("Cannot load model - onnxruntime not installed")
            return False

        try:
            model_path = Path(path)
            if not model_path.exists():
                logger.error(f"Model file not found: {path}")
                return False

            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.inter_op_num_threads = self.config.num_threads

            if self.config.enable_profiling:
                sess_options.enable_profiling = True

            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.config.device == "cuda" else ['CPUExecutionProvider']

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            self.model_loaded = True
            self.model_path = str(path)
            logger.info(f"Loaded threat model from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def unload(self) -> None:
        """Unload the current model."""
        self.session = None
        self.model_loaded = False
        self.model_path = ""
        logger.info("Unloaded threat model")

    def predict(self, features: ThreatFeatures, target_id: str) -> ThreatPrediction:
        """
        Predict threat score for a single target.

        Args:
            features: Extracted threat features
            target_id: ID of the target

        Returns:
            ThreatPrediction with score and confidence
        """
        if not self.model_loaded or self.session is None:
            # Fallback to simple feature-based heuristic
            return self._fallback_predict(features, target_id)

        try:
            # Run inference
            input_data = features.to_numpy().reshape(1, -1)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})

            threat_score = float(outputs[0][0]) * 100.0  # Scale to 0-100
            confidence = float(outputs[1][0]) if len(outputs) > 1 else 0.9

            # Clamp values
            threat_score = max(0.0, min(100.0, threat_score))
            confidence = max(0.0, min(1.0, confidence))

            return ThreatPrediction(
                target_id=target_id,
                threat_score=threat_score,
                confidence=confidence,
                threat_level=self._get_threat_level(threat_score),
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._fallback_predict(features, target_id)

    def predict_batch(
        self,
        features_batch: np.ndarray,
        target_ids: List[str]
    ) -> List[ThreatPrediction]:
        """
        Predict threat scores for multiple targets.

        Args:
            features_batch: numpy array of shape (N, 18)
            target_ids: List of target IDs

        Returns:
            List of ThreatPrediction
        """
        if not self.model_loaded or self.session is None:
            # Fallback for each
            predictions = []
            for i, target_id in enumerate(target_ids):
                feat = ThreatFeatures(*features_batch[i])
                predictions.append(self._fallback_predict(feat, target_id))
            return predictions

        try:
            outputs = self.session.run(
                self.output_names,
                {self.input_name: features_batch.astype(np.float32)}
            )

            threat_scores = outputs[0] * 100.0
            confidences = outputs[1] if len(outputs) > 1 else np.ones(len(target_ids)) * 0.9

            predictions = []
            for i, target_id in enumerate(target_ids):
                score = float(threat_scores[i])
                conf = float(confidences[i])
                predictions.append(ThreatPrediction(
                    target_id=target_id,
                    threat_score=max(0.0, min(100.0, score)),
                    confidence=max(0.0, min(1.0, conf)),
                    threat_level=self._get_threat_level(score),
                ))
            return predictions

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return [self._fallback_predict(
                ThreatFeatures(*features_batch[i]), tid
            ) for i, tid in enumerate(target_ids)]

    def _fallback_predict(self, features: ThreatFeatures, target_id: str) -> ThreatPrediction:
        """
        Fallback prediction when model not available.

        Uses weighted sum of normalized features as proxy for threat score.
        """
        # Weight important features more heavily
        score = (
            (1.0 - features.time_to_impact_normalized) * 35.0 +  # Imminent = high
            features.closing_velocity_normalized * 25.0 +         # Closing = high
            (1.0 - features.aspect_angle_normalized) * 20.0 +     # Head-on = high
            features.altitude_advantage * 10.0 +                  # Above = slightly higher
            features.target_accel_normalized * 10.0               # Maneuvering = harder
        )

        # Normalize to 0-100
        score = max(0.0, min(100.0, score))

        return ThreatPrediction(
            target_id=target_id,
            threat_score=score,
            confidence=0.7,  # Lower confidence for fallback
            threat_level=self._get_threat_level(score),
        )

    @staticmethod
    def _get_threat_level(score: float) -> str:
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"


class GuidanceModel:
    """
    ONNX-based RL guidance policy.

    Input: GuidanceFeatures (24 features)
    Output: acceleration command (3D vector), confidence

    If no model is loaded, falls back to proportional navigation.
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig(model_type="guidance")
        self.session: Optional[Any] = None
        self.input_name: str = "observation"
        self.output_names: List[str] = ["action", "value"]
        self.model_loaded: bool = False
        self.model_path: str = ""
        self.max_accel: float = 100.0  # m/s²

    def load(self, path: str) -> bool:
        """
        Load ONNX policy model from file.

        Args:
            path: Path to .onnx file

        Returns:
            True if loaded successfully
        """
        if not ONNX_AVAILABLE:
            logger.warning("Cannot load model - onnxruntime not installed")
            return False

        try:
            model_path = Path(path)
            if not model_path.exists():
                logger.error(f"Model file not found: {path}")
                return False

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.config.device == "cuda" else ['CPUExecutionProvider']

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            self.model_loaded = True
            self.model_path = str(path)
            logger.info(f"Loaded guidance model from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load guidance model: {e}")
            return False

    def unload(self) -> None:
        """Unload the current model."""
        self.session = None
        self.model_loaded = False
        self.model_path = ""
        logger.info("Unloaded guidance model")

    def predict(self, features: GuidanceFeatures) -> GuidancePrediction:
        """
        Predict guidance command from features.

        Args:
            features: Extracted guidance features

        Returns:
            GuidancePrediction with acceleration command
        """
        if not self.model_loaded or self.session is None:
            return self._fallback_predict(features)

        try:
            input_data = features.to_numpy().reshape(1, -1)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})

            # Assume output is [ax, ay, az] normalized to [-1, 1]
            action = outputs[0][0]
            value = float(outputs[1][0]) if len(outputs) > 1 else 0.5

            # Scale action to acceleration
            accel = Vec3(
                float(action[0]) * self.max_accel,
                float(action[1]) * self.max_accel,
                float(action[2]) * self.max_accel,
            )

            # Clamp magnitude
            mag = accel.magnitude()
            if mag > self.max_accel:
                accel = accel.normalized() * self.max_accel

            return GuidancePrediction(
                acceleration=accel,
                confidence=max(0.0, min(1.0, (value + 1) / 2)),  # value -> confidence
            )

        except Exception as e:
            logger.error(f"Guidance inference error: {e}")
            return self._fallback_predict(features)

    def _fallback_predict(self, features: GuidanceFeatures) -> GuidancePrediction:
        """
        Fallback to simple proportional navigation when model not available.

        Uses the LOS and velocity information in features to compute PN command.
        """
        # Reconstruct LOS direction
        los = Vec3(features.los_angle_x, features.los_angle_y, features.los_angle_z)

        # Reconstruct relative velocity (approximate)
        rel_vel = Vec3(
            features.rel_vel_x * 500.0,  # MAX_VELOCITY
            features.rel_vel_y * 500.0,
            features.rel_vel_z * 500.0
        )

        # Compute LOS rate approximation
        range_m = features.range_normalized * 10000.0  # MAX_RANGE
        if range_m < 1.0:
            range_m = 1.0

        # Perpendicular component of relative velocity
        vel_along_los = los * rel_vel.dot(los)
        vel_perp = rel_vel - vel_along_los
        los_rate = vel_perp / range_m

        # Closing velocity
        closing_vel = features.closing_velocity_normalized * 500.0

        # PN law: a = N * Vc * LOS_rate
        N = 4.0  # Navigation constant
        if closing_vel > 10.0:
            accel = los_rate * (N * closing_vel)
        else:
            # Pure pursuit fallback
            accel = los * self.max_accel

        # Clamp
        mag = accel.magnitude()
        if mag > self.max_accel:
            accel = accel.normalized() * self.max_accel

        return GuidancePrediction(
            acceleration=accel,
            confidence=0.6,  # Lower confidence for fallback
        )


@dataclass
class MLModelRegistry:
    """
    Registry for managing loaded ML models.

    Supports multiple models of each type for A/B testing.
    """
    threat_models: Dict[str, ThreatModel] = field(default_factory=dict)
    guidance_models: Dict[str, GuidanceModel] = field(default_factory=dict)
    active_threat_model: Optional[str] = None
    active_guidance_model: Optional[str] = None

    def load_threat_model(self, model_id: str, path: str, config: Optional[MLConfig] = None) -> bool:
        """Load a threat model and register it."""
        model = ThreatModel(config)
        if model.load(path):
            self.threat_models[model_id] = model
            if self.active_threat_model is None:
                self.active_threat_model = model_id
            return True
        return False

    def load_guidance_model(self, model_id: str, path: str, config: Optional[MLConfig] = None) -> bool:
        """Load a guidance model and register it."""
        model = GuidanceModel(config)
        if model.load(path):
            self.guidance_models[model_id] = model
            if self.active_guidance_model is None:
                self.active_guidance_model = model_id
            return True
        return False

    def unload_threat_model(self, model_id: str) -> bool:
        """Unload a threat model."""
        if model_id in self.threat_models:
            self.threat_models[model_id].unload()
            del self.threat_models[model_id]
            if self.active_threat_model == model_id:
                self.active_threat_model = next(iter(self.threat_models.keys()), None)
            return True
        return False

    def unload_guidance_model(self, model_id: str) -> bool:
        """Unload a guidance model."""
        if model_id in self.guidance_models:
            self.guidance_models[model_id].unload()
            del self.guidance_models[model_id]
            if self.active_guidance_model == model_id:
                self.active_guidance_model = next(iter(self.guidance_models.keys()), None)
            return True
        return False

    def get_active_threat_model(self) -> Optional[ThreatModel]:
        """Get the currently active threat model."""
        if self.active_threat_model:
            return self.threat_models.get(self.active_threat_model)
        return None

    def get_active_guidance_model(self) -> Optional[GuidanceModel]:
        """Get the currently active guidance model."""
        if self.active_guidance_model:
            return self.guidance_models.get(self.active_guidance_model)
        return None

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all registered models."""
        return {
            "threat_models": [
                {
                    "model_id": mid,
                    "path": m.model_path,
                    "loaded": m.model_loaded,
                    "active": mid == self.active_threat_model
                }
                for mid, m in self.threat_models.items()
            ],
            "guidance_models": [
                {
                    "model_id": mid,
                    "path": m.model_path,
                    "loaded": m.model_loaded,
                    "active": mid == self.active_guidance_model
                }
                for mid, m in self.guidance_models.items()
            ]
        }


# Global model registry
_model_registry: Optional[MLModelRegistry] = None


def get_model_registry() -> MLModelRegistry:
    """Get or create the global model registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = MLModelRegistry()
    return _model_registry
