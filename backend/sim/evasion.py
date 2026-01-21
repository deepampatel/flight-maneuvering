"""
Evasion Maneuvers - Target Countermeasures

Real targets don't fly straight - they actively try to defeat interceptors.

EVASION STRATEGIES:

1. CONSTANT TURN
   - Sustained turn at fixed rate
   - Simple but effective against pure pursuit
   - Used by aircraft in defensive BFM

2. WEAVE (S-TURNS)
   - Periodic direction reversals
   - Creates unpredictable path
   - Degrades PN guidance accuracy

3. BARREL ROLL
   - 3D spiral maneuver
   - Combines horizontal and vertical evasion
   - Very challenging for guidance systems

4. RANDOM JINK
   - Random direction changes at intervals
   - Maximizes unpredictability
   - Most difficult to counter

The key insight: evasion works by creating LOS rate changes
that guidance systems can't perfectly track.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable
import math
import random

from .vector import Vec3


class EvasionType(str, Enum):
    NONE = "none"
    CONSTANT_TURN = "constant_turn"
    WEAVE = "weave"
    BARREL_ROLL = "barrel_roll"
    RANDOM_JINK = "random_jink"


@dataclass
class EvasionConfig:
    """Configuration for evasion maneuvers."""
    # Turn rate in degrees per second (for constant_turn)
    turn_rate: float = 15.0

    # Weave parameters
    weave_period: float = 4.0  # seconds for full S-turn cycle
    weave_amplitude: float = 30.0  # max turn rate in deg/s

    # Barrel roll parameters
    roll_rate: float = 20.0  # degrees per second
    roll_pitch_amplitude: float = 15.0  # vertical component

    # Jink parameters
    jink_interval: float = 2.0  # seconds between direction changes
    jink_max_turn: float = 45.0  # max turn angle in degrees

    # General
    max_accel: float = 30.0  # m/s^2 (target G-limit)


@dataclass
class EvasionState:
    """Mutable state for evasion maneuvers."""
    time: float = 0.0
    last_jink_time: float = 0.0
    current_jink_direction: Vec3 = None
    phase: float = 0.0  # For periodic maneuvers

    def __post_init__(self):
        if self.current_jink_direction is None:
            self.current_jink_direction = Vec3.zero()


# Type alias for evasion function
EvasionFunction = Callable[[Vec3, Vec3, float, EvasionState, EvasionConfig], Vec3]


def no_evasion(
    position: Vec3,
    velocity: Vec3,
    dt: float,
    state: EvasionState,
    config: EvasionConfig,
) -> Vec3:
    """No evasion - target flies straight."""
    return Vec3.zero()


def constant_turn(
    position: Vec3,
    velocity: Vec3,
    dt: float,
    state: EvasionState,
    config: EvasionConfig,
) -> Vec3:
    """
    Constant turn maneuver.

    Creates centripetal acceleration perpendicular to velocity.
    Turn is in the horizontal plane.
    """
    speed = velocity.magnitude()
    if speed < 1.0:
        return Vec3.zero()

    # Convert turn rate to radians/second
    omega = math.radians(config.turn_rate)

    # Centripetal acceleration: a = v^2 / r = v * omega
    accel_magnitude = speed * omega

    # Get horizontal velocity direction
    vel_horizontal = Vec3(velocity.x, velocity.y, 0).normalized()

    # Perpendicular direction (90 degrees left in horizontal plane)
    perp = Vec3(-vel_horizontal.y, vel_horizontal.x, 0)

    # Apply turn acceleration
    accel = perp * min(accel_magnitude, config.max_accel)

    return accel


def weave_maneuver(
    position: Vec3,
    velocity: Vec3,
    dt: float,
    state: EvasionState,
    config: EvasionConfig,
) -> Vec3:
    """
    Weave (S-turn) maneuver.

    Sinusoidal turn rate creates periodic direction reversals.
    """
    speed = velocity.magnitude()
    if speed < 1.0:
        return Vec3.zero()

    # Update phase
    state.phase += (2 * math.pi / config.weave_period) * dt

    # Sinusoidal turn rate
    turn_rate_deg = config.weave_amplitude * math.sin(state.phase)
    omega = math.radians(turn_rate_deg)

    # Centripetal acceleration
    accel_magnitude = speed * omega

    # Horizontal perpendicular direction
    vel_horizontal = Vec3(velocity.x, velocity.y, 0).normalized()
    perp = Vec3(-vel_horizontal.y, vel_horizontal.x, 0)

    accel = perp * min(abs(accel_magnitude), config.max_accel) * (1 if accel_magnitude >= 0 else -1)

    return accel


def barrel_roll(
    position: Vec3,
    velocity: Vec3,
    dt: float,
    state: EvasionState,
    config: EvasionConfig,
) -> Vec3:
    """
    Barrel roll maneuver.

    3D spiral combining horizontal turn and vertical oscillation.
    Creates a corkscrew-like path.
    """
    speed = velocity.magnitude()
    if speed < 1.0:
        return Vec3.zero()

    # Update phase for the roll
    roll_omega = math.radians(config.roll_rate)
    state.phase += roll_omega * dt

    # Horizontal component: constant turn
    vel_horizontal = Vec3(velocity.x, velocity.y, 0).normalized()
    perp_h = Vec3(-vel_horizontal.y, vel_horizontal.x, 0)

    # Vertical component: sinusoidal
    pitch_omega = math.radians(config.roll_pitch_amplitude)
    vertical_accel = pitch_omega * speed * math.sin(state.phase)

    # Horizontal turn rate (modulated by roll phase)
    horizontal_accel = config.turn_rate * speed * math.cos(state.phase) * 0.1

    # Combine accelerations
    accel = perp_h * horizontal_accel + Vec3(0, 0, vertical_accel)

    # Clamp to max
    accel_mag = accel.magnitude()
    if accel_mag > config.max_accel:
        accel = accel.normalized() * config.max_accel

    return accel


def random_jink(
    position: Vec3,
    velocity: Vec3,
    dt: float,
    state: EvasionState,
    config: EvasionConfig,
) -> Vec3:
    """
    Random jink maneuver.

    Unpredictable direction changes at random intervals.
    """
    speed = velocity.magnitude()
    if speed < 1.0:
        return Vec3.zero()

    state.time += dt

    # Check if it's time for a new jink
    if state.time - state.last_jink_time >= config.jink_interval:
        state.last_jink_time = state.time

        # Random turn angle
        turn_angle = random.uniform(-config.jink_max_turn, config.jink_max_turn)
        pitch_angle = random.uniform(-15, 15)  # Some vertical variation

        # Convert to direction
        omega_h = math.radians(turn_angle)
        omega_v = math.radians(pitch_angle)

        # Horizontal perpendicular
        vel_horizontal = Vec3(velocity.x, velocity.y, 0).normalized()
        perp_h = Vec3(-vel_horizontal.y, vel_horizontal.x, 0)

        # Compute acceleration direction
        state.current_jink_direction = (
            perp_h * math.sin(omega_h) + Vec3(0, 0, math.sin(omega_v))
        ).normalized()

    # Apply current jink acceleration
    if state.current_jink_direction.magnitude() > 0.1:
        accel = state.current_jink_direction * config.max_accel * 0.8
    else:
        accel = Vec3.zero()

    return accel


# Registry of evasion functions
EVASION_MANEUVERS = {
    EvasionType.NONE: no_evasion,
    EvasionType.CONSTANT_TURN: constant_turn,
    EvasionType.WEAVE: weave_maneuver,
    EvasionType.BARREL_ROLL: barrel_roll,
    EvasionType.RANDOM_JINK: random_jink,
}


def create_evasion_function(
    evasion_type: EvasionType,
    config: EvasionConfig = None,
) -> tuple[EvasionFunction, EvasionState, EvasionConfig]:
    """
    Factory function to create an evasion function with its state.

    Returns:
        (evasion_function, initial_state, config)
    """
    config = config or EvasionConfig()
    state = EvasionState()
    fn = EVASION_MANEUVERS[evasion_type]
    return fn, state, config
