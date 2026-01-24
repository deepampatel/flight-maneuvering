"""
Environmental Effects Model

This module adds atmospheric realism to the simulation:
- Wind (constant + gusts)
- Atmospheric drag
- Air density variation with altitude

Physics background:
- Drag force: F_d = 0.5 * ρ * v² * C_d * A
  where ρ = air density, v = velocity, C_d = drag coefficient, A = cross-section area
- Wind adds to relative velocity for drag calculations
- Air density decreases exponentially with altitude (barometric formula)

All effects are optional and disabled by default for backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from .vector import Vec3


@dataclass
class EnvironmentConfig:
    """
    Configuration for environmental effects.

    All effects disabled by default for backward compatibility.
    """
    # Wind configuration
    wind_velocity: Vec3 = field(default_factory=Vec3.zero)  # m/s in world frame
    wind_gust_amplitude: float = 0.0  # m/s peak gust magnitude
    wind_gust_period: float = 5.0     # seconds for one gust cycle

    # Drag configuration
    enable_drag: bool = False
    reference_drag_coefficient: float = 0.3  # Default Cd for streamlined body

    # Atmosphere configuration
    sea_level_density: float = 1.225   # kg/m³ at sea level (ISA standard)
    scale_height: float = 8500.0       # meters (atmospheric scale height)

    def to_dict(self) -> dict:
        return {
            "wind_velocity": self.wind_velocity.to_dict(),
            "wind_gust_amplitude": self.wind_gust_amplitude,
            "wind_gust_period": self.wind_gust_period,
            "enable_drag": self.enable_drag,
            "reference_drag_coefficient": self.reference_drag_coefficient,
            "sea_level_density": self.sea_level_density,
            "scale_height": self.scale_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentConfig":
        wind_vel = data.get("wind_velocity", {"x": 0, "y": 0, "z": 0})
        return cls(
            wind_velocity=Vec3(wind_vel["x"], wind_vel["y"], wind_vel["z"]),
            wind_gust_amplitude=data.get("wind_gust_amplitude", 0.0),
            wind_gust_period=data.get("wind_gust_period", 5.0),
            enable_drag=data.get("enable_drag", False),
            reference_drag_coefficient=data.get("reference_drag_coefficient", 0.3),
            sea_level_density=data.get("sea_level_density", 1.225),
            scale_height=data.get("scale_height", 8500.0),
        )


@dataclass
class EnvironmentState:
    """
    Time-varying environmental state.

    Caches computed values to avoid redundant calculations within a tick.
    """
    time: float = 0.0
    current_wind: Vec3 = field(default_factory=Vec3.zero)

    # Density cache: altitude -> density (cleared each tick)
    _density_cache: dict = field(default_factory=dict)

    # Pre-computed gust factor (reused for all entities in tick)
    _gust_factor: float = 0.0

    def clear_cache(self) -> None:
        """Clear per-tick caches."""
        self._density_cache.clear()


class EnvironmentModel:
    """
    Computes environmental effects on entities.

    Features:
    - Wind with optional sinusoidal gusts
    - Altitude-dependent air density
    - Aerodynamic drag

    Usage:
        env = EnvironmentModel(config)
        # Each timestep:
        wind = env.get_wind_at(position, time)
        density = env.get_density_at(altitude)
        drag_accel = env.compute_drag_acceleration(velocity, altitude, cross_section, mass)
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self.state = EnvironmentState()

    def update(self, time: float) -> None:
        """
        Update time-varying state (call once per tick).

        This caches values that are reused for all entities in this tick,
        improving performance significantly in multi-entity scenarios.
        """
        self.state.time = time
        self.state.clear_cache()

        # Pre-compute gust factor for this tick (used by all wind queries)
        if self.config.wind_gust_amplitude > 0 and self.config.wind_gust_period > 0:
            self.state._gust_factor = math.sin(2 * math.pi * time / self.config.wind_gust_period)
        else:
            self.state._gust_factor = 0.0

        # Cache the current wind (most entities use uniform wind)
        self.state.current_wind = self._compute_wind_fast()

    def _compute_wind_fast(self) -> Vec3:
        """
        Fast wind computation using pre-cached gust factor.

        Called once per tick in update(), result cached in state.current_wind.
        """
        base_wind = self.config.wind_velocity

        if self.state._gust_factor != 0.0:
            gust_magnitude = self.config.wind_gust_amplitude * self.state._gust_factor

            # Gust aligned with base wind direction (or arbitrary if no base wind)
            if base_wind.magnitude() > 1e-6:
                gust_dir = base_wind.normalized()
            else:
                gust_dir = Vec3(1, 0, 0)  # Default gust direction

            return base_wind + gust_dir * gust_magnitude

        return base_wind

    def get_wind_at(self, position: Vec3, time: float) -> Vec3:
        """
        Get wind velocity at a given position and time.

        Currently uniform wind field - returns cached value from update().
        Future: Could add spatial variation, turbulence models.

        Args:
            position: World position (currently unused, for future spatial variation)
            time: Simulation time in seconds

        Returns:
            Wind velocity vector in m/s
        """
        # For uniform wind, just return the cached value
        # This avoids redundant sin() calculations for each entity
        return self.state.current_wind

    def get_density_at(self, altitude: float) -> float:
        """
        Get air density at a given altitude using barometric formula.

        ρ(h) = ρ₀ * exp(-h / H)

        where:
            ρ₀ = sea level density (1.225 kg/m³)
            H = scale height (~8500m for Earth's atmosphere)
            h = altitude in meters

        Args:
            altitude: Height above sea level in meters (z coordinate if z=up)

        Returns:
            Air density in kg/m³

        Note: Results are cached per-tick at 100m altitude bands for performance.
        """
        # Clamp altitude to reasonable range
        altitude = max(0, altitude)

        # Round to 100m bands for caching (entities at similar altitudes share result)
        # This reduces exp() calls significantly while maintaining <1% accuracy
        alt_key = int(altitude / 100) * 100

        if alt_key in self.state._density_cache:
            return self.state._density_cache[alt_key]

        density = self.config.sea_level_density * math.exp(
            -alt_key / self.config.scale_height
        )

        self.state._density_cache[alt_key] = density
        return density

    def compute_drag_acceleration(
        self,
        velocity: Vec3,
        altitude: float,
        cross_section: float,
        mass: float,
        drag_coefficient: Optional[float] = None,
    ) -> Vec3:
        """
        Compute acceleration due to aerodynamic drag.

        Drag force: F_d = 0.5 * ρ * v² * C_d * A
        Drag acceleration: a_d = F_d / m (opposite to velocity)

        Args:
            velocity: Entity velocity in m/s (world frame)
            altitude: Height for density calculation
            cross_section: Frontal area in m²
            mass: Entity mass in kg
            drag_coefficient: Optional override for Cd

        Returns:
            Drag acceleration vector (opposite to velocity direction)
        """
        if not self.config.enable_drag:
            return Vec3.zero()

        if mass <= 0 or cross_section <= 0:
            return Vec3.zero()

        # Get relative velocity (entity velocity minus wind)
        wind = self.get_wind_at(Vec3.zero(), self.state.time)
        relative_velocity = velocity - wind

        speed = relative_velocity.magnitude()
        if speed < 1e-6:
            return Vec3.zero()

        # Air density at altitude
        rho = self.get_density_at(altitude)

        # Drag coefficient
        cd = drag_coefficient if drag_coefficient is not None else self.config.reference_drag_coefficient

        # Drag force magnitude: F = 0.5 * ρ * v² * Cd * A
        drag_force_magnitude = 0.5 * rho * speed * speed * cd * cross_section

        # Drag acceleration (opposite to relative velocity)
        drag_accel_magnitude = drag_force_magnitude / mass
        drag_direction = -relative_velocity.normalized()

        return drag_direction * drag_accel_magnitude

    def get_total_environmental_acceleration(
        self,
        velocity: Vec3,
        position: Vec3,
        cross_section: float,
        mass: float,
        drag_coefficient: Optional[float] = None,
    ) -> Vec3:
        """
        Get total acceleration from all environmental effects.

        Currently just drag, but could include:
        - Wind-induced forces
        - Gravity variations
        - Other atmospheric effects

        Args:
            velocity: Entity velocity
            position: Entity position (z = altitude)
            cross_section: Frontal area in m²
            mass: Entity mass in kg
            drag_coefficient: Optional Cd override

        Returns:
            Total environmental acceleration
        """
        altitude = max(0, position.z)  # z-up coordinate system

        return self.compute_drag_acceleration(
            velocity=velocity,
            altitude=altitude,
            cross_section=cross_section,
            mass=mass,
            drag_coefficient=drag_coefficient,
        )


def create_wind_from_speed_direction(speed: float, direction_deg: float) -> Vec3:
    """
    Create wind vector from speed and compass direction.

    Args:
        speed: Wind speed in m/s
        direction_deg: Wind direction in degrees (0=North, 90=East, meteorological convention)

    Returns:
        Wind velocity vector
    """
    # Convert to radians (meteorological: direction wind is FROM)
    # We want direction wind is going TO
    direction_rad = math.radians(direction_deg + 180)

    # ENU coordinate system: x=East, y=North
    vx = speed * math.sin(direction_rad)  # East component
    vy = speed * math.cos(direction_rad)  # North component

    return Vec3(vx, vy, 0)
