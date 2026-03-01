"""
Entities: Things that exist in the simulation world.

The key insight here is KINEMATICS - the study of motion:

    position(t+dt) = position(t) + velocity(t) * dt + 0.5 * acceleration(t) * dt²
    velocity(t+dt) = velocity(t) + acceleration(t) * dt

This is just physics 101, but it's the foundation of ALL simulation.

For our interceptor, we control ACCELERATION (thrust/steering).
The sim integrates that into velocity and position over time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import numpy as np

from .vector import Vec3
from .ballistics import (
    FlightPhase,
    PropulsionConfig,
    apply_ballistic_physics,
    determine_flight_phase,
    compute_current_mass,
)


class EntityType(str, Enum):
    TARGET = "target"
    INTERCEPTOR = "interceptor"


@dataclass
class Entity:
    """
    A physical object in 3D space.

    State:
    - position: where it is (meters)
    - velocity: how fast it's moving (m/s)
    - acceleration: current acceleration (m/s²) - set by control/guidance

    Parameters:
    - max_accel: maximum acceleration magnitude (m/s²)
    Physical properties (for environmental effects):
    - mass, cross_section, drag_coefficient

    Ballistic properties (for realistic flight phases):
    - flight_phase: BOOST / BALLISTIC / TERMINAL
    - propulsion: rocket motor config
    - burn_elapsed, dry_mass, guided, threat_type, enable_gravity
    """
    id: str
    entity_type: EntityType
    position: Vec3
    velocity: Vec3
    acceleration: Vec3 = field(default_factory=Vec3.zero)
    max_accel: float = 50.0

    # Physical properties
    mass: float = 150.0
    cross_section: float = 0.05
    drag_coefficient: float = 0.3

    # Ballistic properties
    flight_phase: FlightPhase = FlightPhase.BALLISTIC
    propulsion: PropulsionConfig | None = None
    burn_elapsed: float = 0.0
    dry_mass: float = 150.0
    guided: bool = True
    threat_type: str = ""
    enable_gravity: bool = False
    engagement_decision: str = ""  # 'engage', 'track_only', 'ignore', '' (not evaluated)

    def speed(self) -> float:
        """Current speed (magnitude of velocity)."""
        return self.velocity.magnitude()

    def update(self, dt: float) -> None:
        """
        Integrate motion equations for one timestep.

        Uses semi-implicit Euler:
        1. Apply ballistic physics (gravity, thrust, drag)
        2. Add guidance/evasion acceleration
        3. Clamp total to max_accel
        4. Integrate velocity and position
        """
        # Apply ballistic physics if enabled
        if self.enable_gravity or self.propulsion:
            ballistic_accel, self.burn_elapsed = apply_ballistic_physics(
                velocity=self.velocity,
                position=self.position,
                dry_mass=self.dry_mass,
                cross_section=self.cross_section,
                drag_coefficient=self.drag_coefficient,
                propulsion=self.propulsion,
                burn_elapsed=self.burn_elapsed,
                dt=dt,
                enable_gravity=self.enable_gravity,
                enable_drag=True,
            )
            # Add ballistic forces to commanded acceleration
            self.acceleration = self.acceleration + ballistic_accel

            # Update flight phase
            self.flight_phase = determine_flight_phase(
                self.propulsion, self.burn_elapsed
            )

            # Update mass
            self.mass = compute_current_mass(
                self.dry_mass, self.propulsion, self.burn_elapsed
            )

        # Clamp acceleration to physical limits
        accel_mag = self.acceleration.magnitude()
        if accel_mag > self.max_accel:
            self.acceleration = self.acceleration.normalized() * self.max_accel

        # Semi-implicit Euler integration
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt

        # Ground collision: clamp at z=0
        if self.position.z < 0:
            self.position = Vec3(self.position.x, self.position.y, 0.0)
            self.velocity = Vec3(self.velocity.x, self.velocity.y, 0.0)

    def set_acceleration(self, accel: Vec3) -> None:
        """
        Set commanded acceleration (will be clamped in update).

        In a real system, this comes from the GUIDANCE law.
        """
        self.acceleration = accel

    def to_state_dict(self) -> dict:
        """Serialize current state for transmission."""
        d = {
            "id": self.id,
            "type": self.entity_type.value,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "speed": self.speed(),
            "mass": self.mass,
            "cross_section": self.cross_section,
            "drag_coefficient": self.drag_coefficient,
            # Ballistic fields
            "flight_phase": self.flight_phase.value,
            "guided": self.guided,
            "threat_type": self.threat_type,
            "engagement_decision": self.engagement_decision,
        }
        if self.propulsion:
            fuel_remaining = max(
                0.0,
                self.propulsion.fuel_mass * (1.0 - self.burn_elapsed / self.propulsion.burn_time)
            ) if self.propulsion.burn_time > 0 else 0.0
            d["fuel_remaining"] = fuel_remaining
            d["fuel_total"] = self.propulsion.fuel_mass
            d["burn_time"] = self.propulsion.burn_time
            d["burn_elapsed"] = self.burn_elapsed
        return d


def create_target(
    start_pos: Vec3,
    velocity: Vec3,
    target_id: str = "T1",
    mass: float = 500.0,           # kg (larger aircraft/missile)
    cross_section: float = 0.2,    # m² (larger frontal area)
    drag_coefficient: float = 0.4,  # Less streamlined
) -> Entity:
    """
    Create a target entity.

    Targets in MVP fly with constant velocity (no maneuvering yet).
    Later we'll add turn rates, jinking, etc.
    """
    return Entity(
        id=target_id,
        entity_type=EntityType.TARGET,
        position=start_pos,
        velocity=velocity,
        acceleration=Vec3.zero(),
        max_accel=30.0,  # Targets typically less maneuverable
        mass=mass,
        cross_section=cross_section,
        drag_coefficient=drag_coefficient,
    )


def create_interceptor(
    start_pos: Vec3,
    initial_velocity: Vec3,
    interceptor_id: str = "I1",
    mass: float = 150.0,           # kg (smaller, agile missile)
    cross_section: float = 0.05,   # m² (small frontal area)
    drag_coefficient: float = 0.3,  # Streamlined
) -> Entity:
    """
    Create an interceptor entity.

    The interceptor is what WE control. Its acceleration
    is set by the guidance law each timestep.
    """
    return Entity(
        id=interceptor_id,
        entity_type=EntityType.INTERCEPTOR,
        position=start_pos,
        velocity=initial_velocity,
        acceleration=Vec3.zero(),
        max_accel=100.0,  # Missiles can pull more Gs
        mass=mass,
        cross_section=cross_section,
        drag_coefficient=drag_coefficient,
    )


def create_threat_from_catalog(
    threat_type: str,
    start_pos: Vec3,
    velocity: Vec3,
    target_id: str = "T1",
    enable_gravity: bool = True,
) -> Entity:
    """Create a target from the threat catalog (qassam, grad, etc.)."""
    from .threat_types import THREAT_CATALOG
    profile = THREAT_CATALOG[threat_type]
    return Entity(
        id=target_id,
        entity_type=EntityType.TARGET,
        position=start_pos,
        velocity=velocity,
        acceleration=Vec3.zero(),
        max_accel=profile.max_accel if profile.guided else 0.0,
        mass=profile.total_mass,
        cross_section=profile.cross_section,
        drag_coefficient=profile.drag_coefficient,
        flight_phase=FlightPhase.BOOST,
        propulsion=profile.propulsion,
        burn_elapsed=0.0,
        dry_mass=profile.dry_mass,
        guided=profile.guided,
        threat_type=threat_type,
        enable_gravity=enable_gravity,
    )


def create_interceptor_from_catalog(
    interceptor_type: str,
    start_pos: Vec3,
    velocity: Vec3,
    interceptor_id: str = "I1",
    enable_gravity: bool = True,
) -> Entity:
    """Create an interceptor from the catalog (tamir, stunner, etc.)."""
    from .threat_types import INTERCEPTOR_CATALOG
    profile = INTERCEPTOR_CATALOG[interceptor_type]
    return Entity(
        id=interceptor_id,
        entity_type=EntityType.INTERCEPTOR,
        position=start_pos,
        velocity=velocity,
        acceleration=Vec3.zero(),
        max_accel=profile.max_accel,
        mass=profile.total_mass,
        cross_section=profile.cross_section,
        drag_coefficient=profile.drag_coefficient,
        flight_phase=FlightPhase.BOOST,
        propulsion=profile.propulsion,
        burn_elapsed=0.0,
        dry_mass=profile.dry_mass,
        guided=True,
        threat_type=interceptor_type,
        enable_gravity=enable_gravity,
    )
