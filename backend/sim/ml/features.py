"""
Feature Extraction for ML Models

This module extracts normalized features from simulation state
for use with neural network threat assessment and guidance policies.

FEATURE DESIGN PRINCIPLES:
1. All features normalized to [-1, 1] or [0, 1] range
2. Features should be invariant to absolute position (relative only)
3. Include both kinematic and geometric features
4. Features should capture tactical relevance
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import math

from ..vector import Vec3
from ..entities import Entity
from ..intercept import InterceptGeometry


@dataclass
class ThreatFeatures:
    """
    Feature vector for threat assessment model.

    18 features total, all normalized to reasonable ranges.
    """
    # Geometric features (normalized)
    range_normalized: float          # 0-1 (0=close, 1=far, max ~10km)
    closing_velocity_normalized: float  # -1 to 1 (-1=opening, 1=closing fast)
    aspect_angle_normalized: float   # 0-1 (0=head-on, 1=tail)

    # Time features
    time_to_impact_normalized: float # 0-1 (0=imminent, 1=far)

    # Position features (relative)
    rel_pos_x: float  # -1 to 1
    rel_pos_y: float  # -1 to 1
    rel_pos_z: float  # -1 to 1 (altitude difference)

    # Velocity features (relative)
    rel_vel_x: float  # -1 to 1
    rel_vel_y: float  # -1 to 1
    rel_vel_z: float  # -1 to 1

    # Target state
    target_speed_normalized: float   # 0-1
    target_accel_normalized: float   # 0-1
    target_heading_x: float          # -1 to 1
    target_heading_y: float          # -1 to 1

    # Interceptor state
    interceptor_speed_normalized: float  # 0-1
    interceptor_heading_x: float         # -1 to 1
    interceptor_heading_y: float         # -1 to 1

    # Altitude advantage
    altitude_advantage: float        # -1 to 1 (positive = target above)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.range_normalized,
            self.closing_velocity_normalized,
            self.aspect_angle_normalized,
            self.time_to_impact_normalized,
            self.rel_pos_x,
            self.rel_pos_y,
            self.rel_pos_z,
            self.rel_vel_x,
            self.rel_vel_y,
            self.rel_vel_z,
            self.target_speed_normalized,
            self.target_accel_normalized,
            self.target_heading_x,
            self.target_heading_y,
            self.interceptor_speed_normalized,
            self.interceptor_heading_x,
            self.interceptor_heading_y,
            self.altitude_advantage,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Names of features for debugging/visualization."""
        return [
            "range_normalized",
            "closing_velocity_normalized",
            "aspect_angle_normalized",
            "time_to_impact_normalized",
            "rel_pos_x",
            "rel_pos_y",
            "rel_pos_z",
            "rel_vel_x",
            "rel_vel_y",
            "rel_vel_z",
            "target_speed_normalized",
            "target_accel_normalized",
            "target_heading_x",
            "target_heading_y",
            "interceptor_speed_normalized",
            "interceptor_heading_x",
            "interceptor_heading_y",
            "altitude_advantage",
        ]

    @staticmethod
    def num_features() -> int:
        return 18


@dataclass
class GuidanceFeatures:
    """
    Feature vector for RL guidance policy.

    24 features capturing the full tactical situation.
    """
    # Relative position (normalized to max engagement range)
    rel_pos_x: float  # -1 to 1
    rel_pos_y: float  # -1 to 1
    rel_pos_z: float  # -1 to 1

    # Relative velocity (normalized)
    rel_vel_x: float  # -1 to 1
    rel_vel_y: float  # -1 to 1
    rel_vel_z: float  # -1 to 1

    # Own state (interceptor)
    own_vel_x: float  # -1 to 1
    own_vel_y: float  # -1 to 1
    own_vel_z: float  # -1 to 1
    own_speed_normalized: float  # 0 to 1

    # Target state
    target_vel_x: float  # -1 to 1
    target_vel_y: float  # -1 to 1
    target_vel_z: float  # -1 to 1
    target_accel_x: float  # -1 to 1
    target_accel_y: float  # -1 to 1
    target_accel_z: float  # -1 to 1

    # Geometric features
    range_normalized: float  # 0 to 1
    closing_velocity_normalized: float  # -1 to 1
    los_angle_x: float  # -1 to 1 (LOS unit vector)
    los_angle_y: float  # -1 to 1
    los_angle_z: float  # -1 to 1

    # Time/tactical features
    time_to_impact_normalized: float  # 0 to 1
    aspect_angle_normalized: float    # 0 to 1
    altitude_ratio: float             # -1 to 1

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.rel_pos_x,
            self.rel_pos_y,
            self.rel_pos_z,
            self.rel_vel_x,
            self.rel_vel_y,
            self.rel_vel_z,
            self.own_vel_x,
            self.own_vel_y,
            self.own_vel_z,
            self.own_speed_normalized,
            self.target_vel_x,
            self.target_vel_y,
            self.target_vel_z,
            self.target_accel_x,
            self.target_accel_y,
            self.target_accel_z,
            self.range_normalized,
            self.closing_velocity_normalized,
            self.los_angle_x,
            self.los_angle_y,
            self.los_angle_z,
            self.time_to_impact_normalized,
            self.aspect_angle_normalized,
            self.altitude_ratio,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Names of features for debugging."""
        return [
            "rel_pos_x", "rel_pos_y", "rel_pos_z",
            "rel_vel_x", "rel_vel_y", "rel_vel_z",
            "own_vel_x", "own_vel_y", "own_vel_z", "own_speed_normalized",
            "target_vel_x", "target_vel_y", "target_vel_z",
            "target_accel_x", "target_accel_y", "target_accel_z",
            "range_normalized", "closing_velocity_normalized",
            "los_angle_x", "los_angle_y", "los_angle_z",
            "time_to_impact_normalized", "aspect_angle_normalized",
            "altitude_ratio",
        ]

    @staticmethod
    def num_features() -> int:
        return 24


# Normalization constants
MAX_RANGE = 10000.0       # 10 km
MAX_VELOCITY = 500.0      # 500 m/s
MAX_ACCEL = 100.0         # 100 m/s²
MAX_ALTITUDE = 5000.0     # 5 km
MAX_TTI = 60.0            # 60 seconds


def normalize_range(r: float) -> float:
    """Normalize range to [0, 1]."""
    return min(1.0, max(0.0, r / MAX_RANGE))


def normalize_velocity(v: float) -> float:
    """Normalize velocity to [-1, 1]."""
    return max(-1.0, min(1.0, v / MAX_VELOCITY))


def normalize_accel(a: float) -> float:
    """Normalize acceleration to [-1, 1]."""
    return max(-1.0, min(1.0, a / MAX_ACCEL))


def normalize_position(p: float) -> float:
    """Normalize position component to [-1, 1]."""
    return max(-1.0, min(1.0, p / MAX_RANGE))


def normalize_altitude_delta(delta: float) -> float:
    """Normalize altitude difference to [-1, 1]."""
    return max(-1.0, min(1.0, delta / MAX_ALTITUDE))


def normalize_tti(tti: float) -> float:
    """Normalize time to impact to [0, 1]."""
    if tti < 0:
        return 1.0  # Not closing = far
    return min(1.0, max(0.0, tti / MAX_TTI))


def get_heading_components(velocity: Vec3) -> tuple[float, float]:
    """Get normalized X/Y components of velocity direction."""
    speed = velocity.magnitude()
    if speed < 0.1:
        return 0.0, 0.0
    return velocity.x / speed, velocity.y / speed


def extract_threat_features(
    interceptor: Entity,
    target: Entity,
    geometry: Optional[InterceptGeometry] = None,
) -> ThreatFeatures:
    """
    Extract threat assessment features from interceptor/target pair.

    Args:
        interceptor: The interceptor entity
        target: The target to assess
        geometry: Pre-computed intercept geometry (optional, will compute if None)

    Returns:
        ThreatFeatures ready for model input
    """
    # Compute geometry if not provided
    if geometry is None:
        from ..intercept import compute_intercept_geometry
        geometry = compute_intercept_geometry(interceptor, target)

    # Relative vectors
    rel_pos = target.position - interceptor.position
    rel_vel = target.velocity - interceptor.velocity

    # Normalize position components
    rel_pos_x = normalize_position(rel_pos.x)
    rel_pos_y = normalize_position(rel_pos.y)
    rel_pos_z = normalize_altitude_delta(rel_pos.z)

    # Normalize velocity components
    rel_vel_x = normalize_velocity(rel_vel.x)
    rel_vel_y = normalize_velocity(rel_vel.y)
    rel_vel_z = normalize_velocity(rel_vel.z)

    # Target heading
    tgt_heading_x, tgt_heading_y = get_heading_components(target.velocity)

    # Interceptor heading
    int_heading_x, int_heading_y = get_heading_components(interceptor.velocity)

    # Altitude advantage (positive = target above)
    altitude_delta = target.position.z - interceptor.position.z
    altitude_advantage = normalize_altitude_delta(altitude_delta)

    return ThreatFeatures(
        range_normalized=normalize_range(geometry.los_range),
        closing_velocity_normalized=normalize_velocity(geometry.closing_velocity),
        aspect_angle_normalized=geometry.aspect_angle / 180.0,  # 0-1
        time_to_impact_normalized=normalize_tti(geometry.time_to_intercept),
        rel_pos_x=rel_pos_x,
        rel_pos_y=rel_pos_y,
        rel_pos_z=rel_pos_z,
        rel_vel_x=rel_vel_x,
        rel_vel_y=rel_vel_y,
        rel_vel_z=rel_vel_z,
        target_speed_normalized=min(1.0, target.speed() / MAX_VELOCITY),
        target_accel_normalized=min(1.0, target.acceleration.magnitude() / MAX_ACCEL),
        target_heading_x=tgt_heading_x,
        target_heading_y=tgt_heading_y,
        interceptor_speed_normalized=min(1.0, interceptor.speed() / MAX_VELOCITY),
        interceptor_heading_x=int_heading_x,
        interceptor_heading_y=int_heading_y,
        altitude_advantage=altitude_advantage,
    )


def extract_guidance_features(
    interceptor: Entity,
    target: Entity,
    geometry: Optional[InterceptGeometry] = None,
) -> GuidanceFeatures:
    """
    Extract guidance policy features from interceptor/target pair.

    Args:
        interceptor: The interceptor entity
        target: The target to track
        geometry: Pre-computed intercept geometry (optional)

    Returns:
        GuidanceFeatures ready for model input
    """
    # Compute geometry if not provided
    if geometry is None:
        from ..intercept import compute_intercept_geometry
        geometry = compute_intercept_geometry(interceptor, target)

    # Relative vectors
    rel_pos = target.position - interceptor.position
    rel_vel = target.velocity - interceptor.velocity

    # LOS unit vector
    range_mag = geometry.los_range
    if range_mag > 0.1:
        los_unit = rel_pos / range_mag
    else:
        los_unit = Vec3(1, 0, 0)

    return GuidanceFeatures(
        # Relative position
        rel_pos_x=normalize_position(rel_pos.x),
        rel_pos_y=normalize_position(rel_pos.y),
        rel_pos_z=normalize_altitude_delta(rel_pos.z),

        # Relative velocity
        rel_vel_x=normalize_velocity(rel_vel.x),
        rel_vel_y=normalize_velocity(rel_vel.y),
        rel_vel_z=normalize_velocity(rel_vel.z),

        # Own velocity
        own_vel_x=normalize_velocity(interceptor.velocity.x),
        own_vel_y=normalize_velocity(interceptor.velocity.y),
        own_vel_z=normalize_velocity(interceptor.velocity.z),
        own_speed_normalized=min(1.0, interceptor.speed() / MAX_VELOCITY),

        # Target velocity
        target_vel_x=normalize_velocity(target.velocity.x),
        target_vel_y=normalize_velocity(target.velocity.y),
        target_vel_z=normalize_velocity(target.velocity.z),

        # Target acceleration
        target_accel_x=normalize_accel(target.acceleration.x),
        target_accel_y=normalize_accel(target.acceleration.y),
        target_accel_z=normalize_accel(target.acceleration.z),

        # Geometric
        range_normalized=normalize_range(range_mag),
        closing_velocity_normalized=normalize_velocity(geometry.closing_velocity),
        los_angle_x=los_unit.x,
        los_angle_y=los_unit.y,
        los_angle_z=los_unit.z,

        # Time/tactical
        time_to_impact_normalized=normalize_tti(geometry.time_to_intercept),
        aspect_angle_normalized=geometry.aspect_angle / 180.0,
        altitude_ratio=normalize_altitude_delta(
            target.position.z - interceptor.position.z
        ),
    )


def extract_batch_threat_features(
    interceptor: Entity,
    targets: List[Entity],
    geometries: Optional[List[InterceptGeometry]] = None,
) -> np.ndarray:
    """
    Extract threat features for multiple targets as a batch.

    Args:
        interceptor: The interceptor entity
        targets: List of targets to assess
        geometries: Pre-computed geometries (optional)

    Returns:
        numpy array of shape (num_targets, num_features)
    """
    if geometries is None:
        from ..intercept import compute_intercept_geometry
        geometries = [compute_intercept_geometry(interceptor, t) for t in targets]

    features = []
    for target, geom in zip(targets, geometries):
        feat = extract_threat_features(interceptor, target, geom)
        features.append(feat.to_numpy())

    return np.stack(features, axis=0) if features else np.zeros((0, ThreatFeatures.num_features()))
