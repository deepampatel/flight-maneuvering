"""
Ballistic Physics: Flight phases, propulsion, and gravity.

Real rockets/missiles go through distinct flight phases:

  BOOST:     Motor burning. Thrust > drag + gravity. Accelerating.
  BALLISTIC: Motor burned out. Only gravity + drag. Parabolic arc.
  TERMINAL:  Final phase before impact/intercept. Some missiles
             reignite or steer more aggressively.

Key physics:
  - Gravity: always pulling down at ~9.81 m/s² (altitude-independent for now)
  - Thrust: F = T (constant during burn), decreases mass as fuel consumed
  - Drag:   F = 0.5 * rho * v² * Cd * A (altitude-dependent air density)
  - Mass:   decreases during boost as fuel burns

This is what makes Iron Dome interesting: Qassam rockets are
UNGUIDED ballistic projectiles. Once the motor burns out,
they follow a predictable parabolic trajectory. The system
can predict the impact point and decide whether to engage.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .vector import Vec3


class FlightPhase(str, Enum):
    """Distinct phases of missile/rocket flight."""
    BOOST = "boost"          # Motor burning, thrust active
    BALLISTIC = "ballistic"  # Motor off, gravity + drag only
    TERMINAL = "terminal"    # Final engagement phase


@dataclass
class PropulsionConfig:
    """Rocket motor parameters.

    thrust:           Motor thrust in Newtons
    burn_time:        How long the motor burns (seconds)
    specific_impulse: Motor efficiency (seconds) — higher = more efficient
    fuel_mass:        Mass of fuel (kg) — consumed during burn
    """
    thrust: float          # N
    burn_time: float       # s
    specific_impulse: float = 250.0  # s (typical solid rocket)
    fuel_mass: float = 20.0  # kg


# Gravity vector (constant for now, could be altitude-dependent later)
GRAVITY = Vec3(0.0, 0.0, -9.81)  # m/s² downward


def compute_drag_force(
    velocity: Vec3,
    mass: float,
    cross_section: float,
    drag_coefficient: float,
    altitude: float = 0.0,
) -> Vec3:
    """
    Aerodynamic drag force opposing motion.

    F_drag = -0.5 * rho * |v|² * Cd * A * v_hat

    rho decreases with altitude (exponential atmosphere model).
    """
    speed = velocity.magnitude()
    if speed < 0.1:
        return Vec3.zero()

    # Exponential atmosphere: rho = rho_0 * exp(-h / H)
    rho_0 = 1.225  # sea-level air density kg/m³
    scale_height = 8500.0  # meters
    rho = rho_0 * (2.718281828 ** (-max(altitude, 0) / scale_height))

    drag_magnitude = 0.5 * rho * speed * speed * drag_coefficient * cross_section
    # Drag opposes velocity
    v_hat = velocity.normalized()
    return v_hat * (-drag_magnitude)


def compute_thrust_force(
    velocity: Vec3,
    propulsion: PropulsionConfig,
    burn_elapsed: float,
) -> Vec3:
    """
    Thrust force along velocity direction.

    Active only during BOOST phase (burn_elapsed < burn_time).
    """
    if burn_elapsed >= propulsion.burn_time:
        return Vec3.zero()

    speed = velocity.magnitude()
    if speed < 0.1:
        # If nearly stationary, thrust upward (launch phase)
        return Vec3(0.0, 0.0, propulsion.thrust)

    return velocity.normalized() * propulsion.thrust


def compute_current_mass(
    dry_mass: float,
    propulsion: PropulsionConfig | None,
    burn_elapsed: float,
) -> float:
    """
    Current total mass = dry mass + remaining fuel.

    Mass decreases linearly during burn.
    """
    if propulsion is None:
        return dry_mass

    fuel_remaining = max(
        0.0,
        propulsion.fuel_mass * (1.0 - burn_elapsed / propulsion.burn_time)
    )
    return dry_mass + fuel_remaining


def determine_flight_phase(
    propulsion: PropulsionConfig | None,
    burn_elapsed: float,
    is_terminal: bool = False,
) -> FlightPhase:
    """Determine current flight phase."""
    if is_terminal:
        return FlightPhase.TERMINAL
    if propulsion and burn_elapsed < propulsion.burn_time:
        return FlightPhase.BOOST
    return FlightPhase.BALLISTIC


def apply_ballistic_physics(
    velocity: Vec3,
    position: Vec3,
    dry_mass: float,
    cross_section: float,
    drag_coefficient: float,
    propulsion: PropulsionConfig | None,
    burn_elapsed: float,
    dt: float,
    enable_gravity: bool = True,
    enable_drag: bool = True,
) -> tuple[Vec3, float]:
    """
    Compute net acceleration from ballistic physics.

    Returns (acceleration_vec, new_burn_elapsed).

    This acceleration is ADDED to any guidance/evasion acceleration.
    For unguided rockets, this is the ONLY acceleration.
    """
    forces = Vec3.zero()

    # 1. Gravity (always on for ballistic objects)
    current_mass = compute_current_mass(dry_mass, propulsion, burn_elapsed)
    if enable_gravity:
        forces = forces + GRAVITY * current_mass

    # 2. Thrust (during boost phase)
    if propulsion and burn_elapsed < propulsion.burn_time:
        thrust_force = compute_thrust_force(velocity, propulsion, burn_elapsed)
        forces = forces + thrust_force
        burn_elapsed = min(burn_elapsed + dt, propulsion.burn_time)

    # 3. Drag
    if enable_drag:
        drag_force = compute_drag_force(
            velocity, current_mass, cross_section, drag_coefficient,
            altitude=position.z,
        )
        forces = forces + drag_force

    # F = ma → a = F/m
    if current_mass > 0:
        acceleration = forces * (1.0 / current_mass)
    else:
        acceleration = Vec3.zero()

    return acceleration, burn_elapsed
