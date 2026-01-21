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
                 This represents physical limits (thrust, structural G-limits)
    """
    id: str
    entity_type: EntityType
    position: Vec3
    velocity: Vec3
    acceleration: Vec3 = field(default_factory=Vec3.zero)
    max_accel: float = 50.0  # ~5G, reasonable for a missile

    def speed(self) -> float:
        """Current speed (magnitude of velocity)."""
        return self.velocity.magnitude()

    def update(self, dt: float) -> None:
        """
        Integrate motion equations for one timestep.

        This uses semi-implicit Euler integration:
        1. Update velocity with current acceleration
        2. Update position with NEW velocity

        Why semi-implicit? It's more stable than explicit Euler
        and good enough for our purposes. Real systems use RK4
        or more sophisticated integrators.
        """
        # Clamp acceleration to physical limits
        accel_mag = self.acceleration.magnitude()
        if accel_mag > self.max_accel:
            self.acceleration = self.acceleration.normalized() * self.max_accel

        # Semi-implicit Euler integration
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt

    def set_acceleration(self, accel: Vec3) -> None:
        """
        Set commanded acceleration (will be clamped in update).

        In a real system, this comes from the GUIDANCE law.
        """
        self.acceleration = accel

    def to_state_dict(self) -> dict:
        """Serialize current state for transmission."""
        return {
            "id": self.id,
            "type": self.entity_type.value,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "speed": self.speed(),
        }


def create_target(
    start_pos: Vec3,
    velocity: Vec3,
    target_id: str = "T1"
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
    )


def create_interceptor(
    start_pos: Vec3,
    initial_velocity: Vec3,
    interceptor_id: str = "I1"
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
    )
