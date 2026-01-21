"""
Intercept Geometry - Optimal Engagement Calculations

This module computes the geometric relationships between interceptors and targets
that are critical for engagement decisions:

- Lead pursuit angles: where to aim to hit a moving target
- Aspect angles: whether we're approaching head-on, tail-chase, or beam
- Time-to-intercept: when will we reach the target
- Collision course detection: are we on track to hit

KEY CONCEPTS:

1. ASPECT ANGLE: The angle between target's velocity and the line from target to us
   - 0° = head-on (we see the nose)
   - 90° = beam (we see the side)
   - 180° = tail-chase (we see the tail)

2. ANTENNA TRAIN ANGLE (ATA): Angle off our nose to the target
   - 0° = target directly ahead
   - 90° = target to our side

3. LEAD ANGLE: How far ahead of the target to aim
   - Derived from collision triangle geometry
   - Depends on relative speeds and aspect angle

4. COLLISION COURSE: Constant bearing, decreasing range
   - If LOS rate ≈ 0 and closing velocity > 0, we're on collision course
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import numpy as np

from .vector import Vec3
from .entities import Entity
from .guidance import compute_los_rate, compute_closing_velocity, GuidanceState


@dataclass
class InterceptGeometry:
    """
    Complete intercept geometry for an interceptor-target pair.

    All angles in degrees, distances in meters, speeds in m/s.
    """
    # Identifiers
    interceptor_id: str
    target_id: str

    # Line of Sight data
    los_vector: Vec3              # Unit vector from interceptor to target
    los_range: float              # Distance to target (meters)
    los_rate_magnitude: float     # Angular rate of LOS (rad/s)

    # Approach geometry (degrees)
    aspect_angle: float           # 0=head-on, 180=tail
    antenna_train_angle: float    # Angle off interceptor nose

    # Lead pursuit computation
    lead_angle: float             # Required lead for collision (degrees)
    lead_pursuit_vector: Vec3     # Optimal velocity direction
    collision_course: bool        # True if current heading leads to intercept

    # Time predictions
    time_to_intercept: float      # Estimated TTI (seconds), -1 if not closing
    predicted_miss_distance: float  # Miss at CPA if no maneuver (meters)

    # Closing kinematics
    closing_velocity: float       # Rate of range decrease (m/s)
    relative_velocity: Vec3       # Full relative velocity vector

    # Predicted intercept point
    intercept_point: Optional[Vec3] = None  # Where collision will occur

    def to_dict(self) -> dict:
        """Serialize for JSON transmission."""
        return {
            "interceptor_id": self.interceptor_id,
            "target_id": self.target_id,
            "los_range": float(round(self.los_range, 1)),
            "los_rate_magnitude": float(round(self.los_rate_magnitude, 4)),
            "aspect_angle": float(round(self.aspect_angle, 1)),
            "antenna_train_angle": float(round(self.antenna_train_angle, 1)),
            "lead_angle": float(round(self.lead_angle, 1)),
            "collision_course": bool(self.collision_course),
            "time_to_intercept": float(round(self.time_to_intercept, 2)) if self.time_to_intercept >= 0 else -1,
            "predicted_miss_distance": float(round(self.predicted_miss_distance, 1)),
            "closing_velocity": float(round(self.closing_velocity, 1)),
            "intercept_point": self.intercept_point.to_dict() if self.intercept_point else None,
        }


def compute_aspect_angle(target_vel: Vec3, los_to_interceptor: Vec3) -> float:
    """
    Compute aspect angle - the angle at which we're approaching the target.

    Aspect angle is measured from the target's velocity vector to the
    line from target to interceptor.

    Returns:
        Angle in degrees (0-180):
        - 0° = pure head-on (nose aspect)
        - 90° = beam (side aspect)
        - 180° = pure tail-chase
    """
    target_speed = target_vel.magnitude()
    los_mag = los_to_interceptor.magnitude()

    if target_speed < 1.0 or los_mag < 1.0:
        return 0.0  # Target stationary or too close

    # Unit vectors
    target_heading = target_vel.normalized()
    los_unit = los_to_interceptor.normalized()

    # Dot product gives cosine of angle
    cos_angle = target_heading.dot(los_unit)

    # Clamp to avoid numerical issues with acos
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Convert to degrees
    # Note: aspect is measured from target velocity to LOS
    # So head-on (approaching from front) has target flying TOWARD us
    # which means target_vel points opposite to LOS_to_interceptor
    return math.degrees(math.acos(-cos_angle))


def compute_antenna_train_angle(interceptor_vel: Vec3, los_to_target: Vec3) -> float:
    """
    Compute antenna train angle (ATA) - angle off our nose to the target.

    This tells us how far we need to turn to point at the target.

    Returns:
        Angle in degrees (0-180):
        - 0° = target directly ahead
        - 90° = target to our beam
        - 180° = target behind us
    """
    interceptor_speed = interceptor_vel.magnitude()
    los_mag = los_to_target.magnitude()

    if interceptor_speed < 1.0 or los_mag < 1.0:
        return 0.0  # We're stationary or target too close

    # Unit vectors
    our_heading = interceptor_vel.normalized()
    los_unit = los_to_target.normalized()

    # Dot product gives cosine of angle between our heading and LOS
    cos_angle = our_heading.dot(los_unit)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def compute_lead_pursuit_angle(
    interceptor_speed: float,
    target_speed: float,
    aspect_angle_deg: float
) -> float:
    """
    Compute the classical lead pursuit angle.

    This is derived from the collision triangle - the angle ahead of
    the target we need to aim to achieve an intercept.

    The formula comes from the sine rule applied to the collision triangle:
        sin(lead) / Vt = sin(aspect) / Vi
        lead = arcsin((Vt/Vi) * sin(aspect))

    Args:
        interceptor_speed: Our speed (m/s)
        target_speed: Target speed (m/s)
        aspect_angle_deg: Aspect angle (degrees)

    Returns:
        Lead angle in degrees. Positive means aim ahead of target.
        Returns 0 if no valid solution (target too fast).
    """
    if interceptor_speed < 1.0:
        return 0.0

    speed_ratio = target_speed / interceptor_speed
    aspect_rad = math.radians(aspect_angle_deg)

    # sin(lead) = (Vt/Vi) * sin(aspect)
    sin_lead = speed_ratio * math.sin(aspect_rad)

    # Check if valid (must be <= 1 for arcsin)
    if abs(sin_lead) > 1.0:
        # Target is faster than we can compensate - maximum lead
        return 90.0 if sin_lead > 0 else -90.0

    return math.degrees(math.asin(sin_lead))


def predict_intercept_point(
    interceptor_pos: Vec3,
    interceptor_speed: float,
    target_pos: Vec3,
    target_vel: Vec3,
    max_iterations: int = 20
) -> Tuple[Optional[Vec3], float]:
    """
    Predict where the intercept will occur using iterative refinement.

    Uses Newton-Raphson style iteration:
    1. Guess time to target's current position
    2. Extrapolate where target will be at that time
    3. Compute new time to reach that point
    4. Repeat until converged

    Returns:
        Tuple of (intercept_point, time_to_intercept)
        Returns (None, -1) if no valid intercept found
    """
    if interceptor_speed < 1.0:
        return None, -1.0

    target_speed = target_vel.magnitude()

    # Initial guess: time to current target position
    initial_range = (target_pos - interceptor_pos).magnitude()
    time_guess = initial_range / interceptor_speed

    for _ in range(max_iterations):
        # Where will target be at time_guess?
        future_target_pos = target_pos + target_vel * time_guess

        # How long to reach that point?
        range_to_future = (future_target_pos - interceptor_pos).magnitude()
        new_time = range_to_future / interceptor_speed

        # Check convergence
        if abs(new_time - time_guess) < 0.01:  # 10ms tolerance
            return future_target_pos, new_time

        time_guess = new_time

        # Sanity check - don't predict too far ahead
        if time_guess > 300:  # 5 minutes max
            return None, -1.0

    # Didn't converge but return best estimate
    future_target_pos = target_pos + target_vel * time_guess
    return future_target_pos, time_guess


def compute_collision_triangle(
    interceptor_pos: Vec3,
    interceptor_speed: float,
    target_pos: Vec3,
    target_vel: Vec3
) -> Tuple[Vec3, float]:
    """
    Solve the collision triangle to find optimal heading.

    The collision triangle has:
    - One side: target's velocity vector (scaled by time)
    - One side: interceptor's velocity vector (scaled by time)
    - One side: initial LOS

    We find the interceptor heading that closes the triangle.

    Returns:
        Tuple of (optimal_heading_unit_vector, time_to_intercept)
    """
    intercept_point, tti = predict_intercept_point(
        interceptor_pos, interceptor_speed, target_pos, target_vel
    )

    if intercept_point is None:
        # Fall back to pure pursuit
        los = target_pos - interceptor_pos
        return los.normalized(), -1.0

    # Optimal heading is toward the intercept point
    heading = (intercept_point - interceptor_pos).normalized()

    return heading, tti


def compute_predicted_miss_distance(
    interceptor_pos: Vec3,
    interceptor_vel: Vec3,
    target_pos: Vec3,
    target_vel: Vec3,
    max_time: float = 60.0
) -> float:
    """
    Compute the miss distance at closest point of approach (CPA).

    This assumes both vehicles continue on their current trajectories
    with no maneuvering.

    Uses quadratic formula to find time of minimum distance.
    """
    # Relative position and velocity
    rel_pos = target_pos - interceptor_pos  # P = Pt - Pi
    rel_vel = target_vel - interceptor_vel  # V = Vt - Vi

    # Distance squared as function of time:
    # D²(t) = |P + V*t|² = |P|² + 2*(P·V)*t + |V|²*t²
    # This is minimized when dD²/dt = 0:
    # 2*(P·V) + 2*|V|²*t = 0
    # t_cpa = -(P·V) / |V|²

    vel_squared = rel_vel.dot(rel_vel)

    if vel_squared < 0.01:  # Effectively no relative motion
        return rel_pos.magnitude()

    t_cpa = -rel_pos.dot(rel_vel) / vel_squared

    # Clamp to reasonable range
    t_cpa = max(0.0, min(t_cpa, max_time))

    # Position at CPA
    cpa_pos = rel_pos + rel_vel * t_cpa

    return cpa_pos.magnitude()


def compute_intercept_geometry(
    interceptor: Entity,
    target: Entity,
    dt: float = 0.02
) -> InterceptGeometry:
    """
    Compute complete intercept geometry for an interceptor-target pair.

    This is the main function that computes all geometric parameters
    needed for engagement decisions.

    Args:
        interceptor: Interceptor entity
        target: Target entity
        dt: Time step for rate calculations

    Returns:
        InterceptGeometry with all computed parameters
    """
    # Basic vectors
    los = target.position - interceptor.position
    los_range = los.magnitude()
    los_unit = los.normalized() if los_range > 0.1 else Vec3(1, 0, 0)
    los_to_interceptor = -los  # From target to interceptor

    # Relative velocity
    rel_vel = target.velocity - interceptor.velocity

    # Create guidance state to reuse existing functions
    guidance_state = GuidanceState(
        interceptor_pos=interceptor.position,
        interceptor_vel=interceptor.velocity,
        interceptor_max_accel=interceptor.max_accel,
        target_pos=target.position,
        target_vel=target.velocity,
        target_accel=target.acceleration,
        dt=dt
    )

    # LOS rate (angular velocity)
    los_rate_vec = compute_los_rate(guidance_state)
    los_rate_mag = los_rate_vec.magnitude()

    # Closing velocity (positive = closing)
    closing_vel = compute_closing_velocity(guidance_state)

    # Aspect angle
    aspect = compute_aspect_angle(target.velocity, los_to_interceptor)

    # Antenna train angle
    ata = compute_antenna_train_angle(interceptor.velocity, los)

    # Lead angle
    lead = compute_lead_pursuit_angle(
        interceptor.speed(),
        target.speed(),
        aspect
    )

    # Optimal heading and intercept point
    lead_heading, tti = compute_collision_triangle(
        interceptor.position,
        interceptor.speed(),
        target.position,
        target.velocity
    )

    # Intercept point prediction
    intercept_point, _ = predict_intercept_point(
        interceptor.position,
        interceptor.speed(),
        target.position,
        target.velocity
    )

    # Predicted miss distance (if no maneuver)
    miss_dist = compute_predicted_miss_distance(
        interceptor.position,
        interceptor.velocity,
        target.position,
        target.velocity
    )

    # Collision course check:
    # On collision course if LOS rate is low AND we're closing
    # Threshold: LOS rate < 0.01 rad/s (about 0.5 deg/s)
    # Convert to Python bool to avoid numpy.bool_ serialization issues
    collision_course = bool(los_rate_mag < 0.02 and closing_vel > 10.0)

    # Time to intercept from closing velocity (simple estimate)
    if closing_vel > 1.0 and tti < 0:
        tti = los_range / closing_vel

    return InterceptGeometry(
        interceptor_id=interceptor.id,
        target_id=target.id,
        los_vector=los_unit,
        los_range=los_range,
        los_rate_magnitude=los_rate_mag,
        aspect_angle=aspect,
        antenna_train_angle=ata,
        lead_angle=lead,
        lead_pursuit_vector=lead_heading,
        collision_course=collision_course,
        time_to_intercept=tti,
        predicted_miss_distance=miss_dist,
        closing_velocity=closing_vel,
        relative_velocity=rel_vel,
        intercept_point=intercept_point
    )


def compute_all_geometries(
    interceptors: list[Entity],
    target: Entity,
    dt: float = 0.02
) -> list[InterceptGeometry]:
    """
    Compute intercept geometry for all interceptor-target pairs.

    Args:
        interceptors: List of interceptor entities
        target: Target entity
        dt: Time step

    Returns:
        List of InterceptGeometry, one per interceptor
    """
    return [
        compute_intercept_geometry(interceptor, target, dt)
        for interceptor in interceptors
    ]
