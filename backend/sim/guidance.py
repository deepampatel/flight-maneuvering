"""
Guidance Laws - The Brains Behind Intercept

This module contains different guidance strategies for the interceptor.

KEY CONCEPT: Line of Sight (LOS)

The Line of Sight is the vector from interceptor to target.
- LOS angle: direction to target
- LOS rate: how fast that direction is changing

If LOS rate = 0, you're on a collision course (constant bearing).
Most guidance laws try to drive LOS rate to zero.

GUIDANCE LAW COMPARISON:

1. PURE PURSUIT
   - Points directly at target
   - Simple but inefficient (curved path)
   - Poor against maneuvering targets
   - Used in: early missiles, some drones

2. PROPORTIONAL NAVIGATION (PN)
   - Commands turn rate proportional to LOS rate
   - More efficient (straighter path)
   - Industry standard for missiles
   - Formula: a = N * Vc * LOS_rate

3. AUGMENTED PN (APN)
   - Adds target acceleration term
   - Better against maneuvering targets
   - Formula: a = N * Vc * LOS_rate + N/2 * At

4. TRUE PN vs PURE PN
   - True PN: acceleration perpendicular to LOS
   - Pure PN: acceleration perpendicular to velocity
   - True PN is more common in practice
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from .vector import Vec3


class GuidanceType(str, Enum):
    PURE_PURSUIT = "pure_pursuit"
    PROPORTIONAL_NAV = "proportional_nav"
    AUGMENTED_PN = "augmented_pn"
    ML_POLICY = "ml_policy"


@dataclass
class GuidanceState:
    """
    State information needed by guidance laws.

    This bundles all the kinematic info the guidance computer needs.
    In a real system, this comes from sensors (radar, IR seeker, etc).
    """
    # Interceptor state
    interceptor_pos: Vec3
    interceptor_vel: Vec3
    interceptor_max_accel: float

    # Target state (from sensors/tracking)
    target_pos: Vec3
    target_vel: Vec3
    target_accel: Vec3 = None  # For augmented PN

    # Previous state (for computing rates)
    prev_los: Optional[Vec3] = None
    dt: float = 0.02

    def __post_init__(self):
        if self.target_accel is None:
            self.target_accel = Vec3.zero()


@dataclass
class GuidanceParams:
    """Parameters that tune guidance behavior."""
    # Navigation constant (N) - typically 3-5
    # Higher N = more aggressive maneuvering
    nav_constant: float = 4.0

    # Minimum range to apply guidance (avoid singularities)
    min_range: float = 1.0

    # Bias term for augmented PN
    use_augmented: bool = False


def compute_los_rate(state: GuidanceState) -> Vec3:
    """
    Compute the Line of Sight rate (angular velocity of LOS vector).

    This is the KEY input to proportional navigation.

    Math derivation:
    LOS = target_pos - interceptor_pos
    LOS_rate = d(LOS)/dt = target_vel - interceptor_vel (relative velocity)

    But we want ANGULAR rate, not linear rate.
    Angular rate = (LOS x LOS_dot) / |LOS|^2

    For 3D, we use the component perpendicular to LOS.
    """
    # Current LOS vector
    los = state.target_pos - state.interceptor_pos
    los_mag = los.magnitude()

    if los_mag < state.dt:  # Too close, avoid division by zero
        return Vec3.zero()

    # LOS unit vector
    los_unit = los / los_mag

    # Relative velocity
    rel_vel = state.target_vel - state.interceptor_vel

    # Component of relative velocity perpendicular to LOS
    # This gives us the angular rate of LOS
    vel_along_los = los_unit * rel_vel.dot(los_unit)
    vel_perp_los = rel_vel - vel_along_los

    # Angular rate = perpendicular velocity / range
    los_rate = vel_perp_los / los_mag

    return los_rate


def compute_closing_velocity(state: GuidanceState) -> float:
    """
    Compute closing velocity (rate of range decrease).

    Positive = closing (getting closer)
    Negative = opening (getting farther)
    """
    los = state.target_pos - state.interceptor_pos
    los_mag = los.magnitude()

    if los_mag < 1.0:
        return 0.0

    los_unit = los / los_mag
    rel_vel = state.interceptor_vel - state.target_vel

    # Closing velocity is relative velocity component along LOS
    return rel_vel.dot(los_unit)


def pure_pursuit(state: GuidanceState, params: GuidanceParams) -> Vec3:
    """
    Pure Pursuit guidance: point directly at target.

    Simple but not optimal. Creates curved pursuit path.
    """
    los = state.target_pos - state.interceptor_pos
    distance = los.magnitude()

    if distance < params.min_range:
        return Vec3.zero()

    direction = los.normalized()
    return direction * state.interceptor_max_accel


def proportional_navigation(state: GuidanceState, params: GuidanceParams) -> Vec3:
    """
    Proportional Navigation (PN): the industry standard.

    The guidance law:
        a_cmd = N * Vc * LOS_rate

    Where:
        N = Navigation constant (3-5 typical)
        Vc = Closing velocity
        LOS_rate = Line of sight angular rate

    This produces acceleration PERPENDICULAR to the LOS,
    which drives the LOS rate to zero (constant bearing = collision).

    Why it works:
    - If target moves right, LOS rotates right
    - PN commands acceleration right to "lead" the target
    - Result: interceptor arrives where target will be
    """
    los = state.target_pos - state.interceptor_pos
    distance = los.magnitude()

    if distance < params.min_range:
        return Vec3.zero()

    # Compute LOS rate
    los_rate = compute_los_rate(state)

    # Compute closing velocity
    closing_vel = compute_closing_velocity(state)

    # If opening (not closing), fall back to pursuit
    if closing_vel < 10.0:  # Threshold to avoid issues
        return pure_pursuit(state, params)

    # PN guidance law: a = N * Vc * LOS_rate
    # The LOS_rate is already a vector perpendicular to LOS
    accel_cmd = los_rate * (params.nav_constant * closing_vel)

    # Augmented PN: add target acceleration compensation
    if params.use_augmented and state.target_accel.magnitude() > 0.1:
        # Add term to compensate for target acceleration
        accel_cmd = accel_cmd + state.target_accel * (params.nav_constant / 2.0)

    # Clamp to max acceleration
    accel_mag = accel_cmd.magnitude()
    if accel_mag > state.interceptor_max_accel:
        accel_cmd = accel_cmd.normalized() * state.interceptor_max_accel

    return accel_cmd


def augmented_pn(state: GuidanceState, params: GuidanceParams) -> Vec3:
    """
    Augmented Proportional Navigation: better against maneuvering targets.

    Adds a term to account for target acceleration.
    """
    params_aug = GuidanceParams(
        nav_constant=params.nav_constant,
        min_range=params.min_range,
        use_augmented=True
    )
    return proportional_navigation(state, params_aug)


# -----------------------------------------------------------------------------
# ML-Based Guidance
# -----------------------------------------------------------------------------

def ml_guidance(state: GuidanceState, params: GuidanceParams, model=None) -> Vec3:
    """
    ML Policy guidance: use trained RL policy for guidance commands.

    If model not available, falls back to proportional navigation.

    Args:
        state: Current guidance state
        params: Guidance parameters
        model: GuidanceModel instance (optional)

    Returns:
        Acceleration command vector
    """
    if model is None:
        # Fallback to PN
        return proportional_navigation(state, params)

    # Import here to avoid circular imports
    from .ml.features import extract_guidance_features
    from .entities import Entity

    # Create temporary entities for feature extraction
    # (in real usage, pass actual entities)
    interceptor = Entity(
        id="temp_interceptor",
        entity_type="interceptor",
        position=state.interceptor_pos,
        velocity=state.interceptor_vel,
        max_accel=state.interceptor_max_accel,
    )
    target = Entity(
        id="temp_target",
        entity_type="target",
        position=state.target_pos,
        velocity=state.target_vel,
        acceleration=state.target_accel,
    )

    # Extract features
    features = extract_guidance_features(interceptor, target)

    # Get ML prediction
    prediction = model.predict(features)

    return prediction.acceleration


# Guidance function registry
GUIDANCE_LAWS = {
    GuidanceType.PURE_PURSUIT: pure_pursuit,
    GuidanceType.PROPORTIONAL_NAV: proportional_navigation,
    GuidanceType.AUGMENTED_PN: augmented_pn,
    # ML_POLICY handled specially in create_guidance_function
}


def create_guidance_function(
    guidance_type: GuidanceType,
    params: Optional[GuidanceParams] = None,
    ml_model=None,
):
    """
    Factory function to create a guidance callable for the sim engine.

    Returns a function with signature: (SimState) -> Vec3

    Args:
        guidance_type: Type of guidance law to use
        params: Guidance parameters
        ml_model: GuidanceModel for ML_POLICY type (optional)
    """
    from .engine import SimState  # Import here to avoid circular

    params = params or GuidanceParams()

    # Handle ML guidance specially
    if guidance_type == GuidanceType.ML_POLICY:
        def ml_guidance_wrapper(sim_state: SimState) -> Vec3:
            """Wrapper for ML guidance policy."""
            state = GuidanceState(
                interceptor_pos=sim_state.interceptor.position,
                interceptor_vel=sim_state.interceptor.velocity,
                interceptor_max_accel=sim_state.interceptor.max_accel,
                target_pos=sim_state.target.position,
                target_vel=sim_state.target.velocity,
                target_accel=sim_state.target.acceleration,
                dt=0.02,
            )
            return ml_guidance(state, params, ml_model)
        return ml_guidance_wrapper

    # Standard guidance laws
    guidance_fn = GUIDANCE_LAWS[guidance_type]

    def guidance(sim_state: SimState) -> Vec3:
        """Wrapper that extracts state and calls guidance law."""
        state = GuidanceState(
            interceptor_pos=sim_state.interceptor.position,
            interceptor_vel=sim_state.interceptor.velocity,
            interceptor_max_accel=sim_state.interceptor.max_accel,
            target_pos=sim_state.target.position,
            target_vel=sim_state.target.velocity,
            target_accel=sim_state.target.acceleration,
            dt=0.02,  # Will be updated by engine
        )
        return guidance_fn(state, params)

    return guidance


def create_ml_guidance_function(model, params: Optional[GuidanceParams] = None):
    """
    Create a guidance function using ML policy model.

    Convenience function that wraps create_guidance_function with ML_POLICY type.

    Args:
        model: GuidanceModel instance
        params: Guidance parameters (optional)

    Returns:
        Callable guidance function
    """
    return create_guidance_function(GuidanceType.ML_POLICY, params, ml_model=model)
