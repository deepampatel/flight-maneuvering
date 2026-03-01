"""
Impact Point Prediction (IPP) — Where will this threat land?

This is the KEY INNOVATION of Iron Dome:
  1. Track the incoming rocket
  2. Predict where it will impact
  3. If it's heading for a populated area → ENGAGE
  4. If it's heading for an open field → LET IT GO (save ammo!)

IPP works by forward-propagating the current state through
ballistic physics (gravity + drag) until the object hits the
ground (z <= 0). This gives us:
  - Predicted impact point (x, y)
  - Time to impact
  - Whether it threatens a protected area

For UNGUIDED rockets (Qassam, Grad), this prediction is very
accurate because the trajectory is deterministic after burnout.
For GUIDED threats, it's less certain (they can change course).
"""

from __future__ import annotations
from dataclasses import dataclass
import math

from .vector import Vec3
from .ballistics import (
    PropulsionConfig,
    apply_ballistic_physics,
    GRAVITY,
)


@dataclass
class ProtectedArea:
    """A defended zone (city, base, infrastructure)."""
    id: str
    name: str
    center: Vec3          # Center point (meters)
    radius: float         # Defense radius (meters)
    priority: int = 1     # 1 = highest priority
    population: int = 0   # For display only


@dataclass
class ImpactPrediction:
    """Result of Impact Point Prediction."""
    threat_id: str
    impact_point: Vec3          # Predicted ground impact (z=0)
    time_to_impact: float       # Seconds until impact
    threatens_area: str | None  # ID of threatened ProtectedArea, or None
    area_name: str | None       # Name of threatened area
    distance_to_area: float     # Distance from impact to nearest area center
    confidence: float           # 0-1, higher for ballistic (unguided) threats
    engage: bool                # Should we intercept this?


def predict_impact_point(
    position: Vec3,
    velocity: Vec3,
    dry_mass: float = 25.0,
    cross_section: float = 0.015,
    drag_coefficient: float = 0.5,
    propulsion: PropulsionConfig | None = None,
    burn_elapsed: float = 999.0,  # Default: motor already burned out
    guided: bool = False,
    max_time: float = 120.0,
    dt: float = 0.1,  # Coarser timestep for prediction (faster)
) -> tuple[Vec3, float]:
    """
    Forward-propagate ballistic trajectory until ground impact.

    Returns (impact_point, time_to_impact).
    If object never hits ground within max_time, returns current projection.
    """
    pos = Vec3(position.x, position.y, position.z)
    vel = Vec3(velocity.x, velocity.y, velocity.z)
    t = 0.0
    burn = burn_elapsed

    while t < max_time:
        # Get ballistic acceleration
        accel, burn = apply_ballistic_physics(
            velocity=vel,
            position=pos,
            dry_mass=dry_mass,
            cross_section=cross_section,
            drag_coefficient=drag_coefficient,
            propulsion=propulsion,
            burn_elapsed=burn,
            dt=dt,
            enable_gravity=True,
            enable_drag=True,
        )

        # Integrate
        vel = vel + accel * dt
        pos = pos + vel * dt
        t += dt

        # Ground impact
        if pos.z <= 0:
            # Interpolate exact impact time
            # pos.z was positive last step, now negative
            frac = (pos.z - 0) / (vel.z * dt) if vel.z != 0 else 0
            t -= frac * dt
            pos = Vec3(pos.x, pos.y, 0.0)
            return pos, t

    # Never hit ground (going up or very high altitude)
    return pos, max_time


def check_threat_to_areas(
    threat_id: str,
    position: Vec3,
    velocity: Vec3,
    dry_mass: float,
    cross_section: float,
    drag_coefficient: float,
    propulsion: PropulsionConfig | None,
    burn_elapsed: float,
    guided: bool,
    protected_areas: list[ProtectedArea],
) -> ImpactPrediction:
    """
    Predict impact and check if it threatens any protected area.

    The engagement decision:
    - Unguided + threatens area → ENGAGE (high confidence)
    - Unguided + open field → IGNORE (save ammo)
    - Guided → ENGAGE always (unpredictable path)
    """
    impact, tti = predict_impact_point(
        position=position,
        velocity=velocity,
        dry_mass=dry_mass,
        cross_section=cross_section,
        drag_coefficient=drag_coefficient,
        propulsion=propulsion,
        burn_elapsed=burn_elapsed,
        guided=guided,
    )

    # Confidence: unguided ballistic = very predictable
    confidence = 0.95 if not guided else 0.5

    # Check against each protected area
    nearest_area_id = None
    nearest_area_name = None
    min_distance = float('inf')

    for area in protected_areas:
        dx = impact.x - area.center.x
        dy = impact.y - area.center.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < min_distance:
            min_distance = distance
            nearest_area_id = area.id
            nearest_area_name = area.name

    # Engagement decision
    threatens = None
    area_name = None
    engage = False

    for area in protected_areas:
        dx = impact.x - area.center.x
        dy = impact.y - area.center.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= area.radius:
            threatens = area.id
            area_name = area.name
            engage = True
            min_distance = distance
            break

    # Guided threats always engaged (unpredictable)
    if guided:
        engage = True

    return ImpactPrediction(
        threat_id=threat_id,
        impact_point=impact,
        time_to_impact=tti,
        threatens_area=threatens,
        area_name=area_name,
        distance_to_area=min_distance,
        confidence=confidence,
        engage=engage,
    )


def compute_pk(
    miss_distance: float,
    lethal_radius: float = 20.0,
) -> float:
    """
    Probability of kill based on miss distance at closest point of approach.

    Uses a Gaussian warhead lethality model:
        Pk = exp(-(miss_distance / lethal_radius)^2)

    At miss_distance = 0:               Pk = 1.0   (direct hit)
    At miss_distance = lethal_radius:    Pk = 0.368 (1/e)
    At miss_distance = 2*lethal_radius:  Pk = 0.018

    Args:
        miss_distance: Distance between interceptor and target at CPA (meters)
        lethal_radius: Effective warhead blast/fragmentation radius (meters)
    """
    import math
    if miss_distance <= 0:
        return 1.0
    return math.exp(-(miss_distance / lethal_radius) ** 2)
