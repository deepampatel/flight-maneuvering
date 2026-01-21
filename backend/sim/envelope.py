"""
Engagement Envelope Analysis - Where Can We Intercept?

The engagement envelope defines the region in space where
intercepts are possible. It's a function of:

- Range to target
- Bearing angle (horizontal angle to target)
- Elevation angle (vertical angle)
- Closing velocity
- Target maneuverability

This module sweeps across these parameters to find the envelope
boundary - the limit of our weapon system's reach.

KEY OUTPUTS:
1. 2D Heatmap: Range vs Bearing with intercept probability
2. 3D Surface: Range x Bearing x Elevation
"""

from __future__ import annotations
import asyncio
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from .vector import Vec3
from .engine import SimEngine, SimConfig, EngagementResult
from .guidance import GuidanceType, GuidanceParams, create_guidance_function
from .evasion import EvasionType, EvasionConfig


@dataclass
class EnvelopeConfig:
    """Configuration for engagement envelope analysis."""
    # Range sweep (meters)
    range_min: float = 1000.0
    range_max: float = 5000.0
    range_steps: int = 10

    # Bearing sweep (degrees from forward, -180 to 180)
    bearing_min: float = -90.0
    bearing_max: float = 90.0
    bearing_steps: int = 10

    # Elevation sweep (degrees, negative = below)
    elevation_min: float = -30.0
    elevation_max: float = 30.0
    elevation_steps: int = 5

    # Statistical confidence
    runs_per_point: int = 10

    # Simulation parameters
    guidance_type: GuidanceType = GuidanceType.PROPORTIONAL_NAV
    nav_constant: float = 4.0
    kill_radius: float = 50.0
    max_time: float = 60.0

    # Target parameters
    target_speed: float = 100.0  # m/s
    evasion_type: EvasionType = EvasionType.NONE

    # Interceptor parameters (fixed position at origin)
    interceptor_speed: float = 200.0  # m/s


@dataclass
class EnvelopePoint:
    """Result for a single point in the envelope."""
    range_m: float
    bearing_deg: float
    elevation_deg: float
    intercept_rate: float
    mean_miss_distance: float
    mean_time_to_intercept: float
    num_runs: int


@dataclass
class EnvelopeResults:
    """Complete engagement envelope results."""
    config: Dict[str, Any]
    points: List[EnvelopePoint]

    # Grid dimensions for reconstruction
    range_values: List[float] = field(default_factory=list)
    bearing_values: List[float] = field(default_factory=list)
    elevation_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON response."""
        # Create 2D heatmap data (range x bearing, averaged over elevation)
        heatmap_2d = self._create_2d_heatmap()

        # Create 3D surface data
        surface_3d = self._create_3d_surface()

        return {
            "config": self.config,
            "range_values": self.range_values,
            "bearing_values": self.bearing_values,
            "elevation_values": self.elevation_values,
            "heatmap_2d": heatmap_2d,
            "surface_3d": surface_3d,
            "points": [
                {
                    "range": p.range_m,
                    "bearing": p.bearing_deg,
                    "elevation": p.elevation_deg,
                    "intercept_rate": p.intercept_rate,
                    "mean_miss_distance": p.mean_miss_distance,
                    "mean_time_to_intercept": p.mean_time_to_intercept,
                }
                for p in self.points
            ],
        }

    def _create_2d_heatmap(self) -> Dict[str, Any]:
        """Create 2D heatmap (range x bearing, max over elevation)."""
        # Initialize grid
        grid = {}
        for p in self.points:
            key = (p.range_m, p.bearing_deg)
            if key not in grid:
                grid[key] = []
            grid[key].append(p.intercept_rate)

        # Take max intercept rate over elevations for each (range, bearing)
        heatmap = []
        for r in self.range_values:
            row = []
            for b in self.bearing_values:
                key = (r, b)
                if key in grid:
                    row.append(max(grid[key]))
                else:
                    row.append(0.0)
            heatmap.append(row)

        return {
            "data": heatmap,
            "x_label": "Bearing (deg)",
            "y_label": "Range (m)",
            "x_values": self.bearing_values,
            "y_values": self.range_values,
        }

    def _create_3d_surface(self) -> Dict[str, Any]:
        """Create 3D surface data (for Three.js visualization)."""
        # Create vertices for 3D surface at each elevation
        surfaces = []

        for elev in self.elevation_values:
            vertices = []
            for r in self.range_values:
                for b in self.bearing_values:
                    # Find matching point
                    point = next(
                        (p for p in self.points
                         if p.range_m == r and p.bearing_deg == b and p.elevation_deg == elev),
                        None
                    )
                    if point:
                        # Convert polar to Cartesian for visualization
                        # x = range * cos(bearing), z = range * sin(bearing)
                        # y = intercept_rate (height)
                        x = r * math.cos(math.radians(b)) / 1000  # Scale to km
                        z = r * math.sin(math.radians(b)) / 1000
                        y = point.intercept_rate  # 0-1

                        vertices.append({
                            "x": x,
                            "y": y,
                            "z": z,
                            "range": r,
                            "bearing": b,
                            "elevation": elev,
                            "intercept_rate": point.intercept_rate,
                        })

            surfaces.append({
                "elevation": elev,
                "vertices": vertices,
            })

        return {"surfaces": surfaces}


async def run_envelope_point(
    config: EnvelopeConfig,
    range_m: float,
    bearing_deg: float,
    elevation_deg: float,
) -> EnvelopePoint:
    """
    Run Monte Carlo simulations for a single envelope point.

    The target starts at (range_m, bearing_deg, elevation_deg) from origin
    and flies toward the interceptor's initial position.
    """
    # Convert polar to Cartesian
    # Bearing: 0 = east (positive X), 90 = north (positive Y)
    # Elevation: positive = above horizon
    bearing_rad = math.radians(bearing_deg)
    elevation_rad = math.radians(elevation_deg)

    # Target position
    horizontal_dist = range_m * math.cos(elevation_rad)
    target_x = horizontal_dist * math.cos(bearing_rad)
    target_y = horizontal_dist * math.sin(bearing_rad)
    target_z = range_m * math.sin(elevation_rad) + 800  # Base altitude 800m

    target_start = Vec3(target_x, target_y, target_z)

    # Target velocity: flying toward origin (interceptor)
    direction_to_origin = Vec3(-target_x, -target_y, -target_z + 600).normalized()
    target_velocity = direction_to_origin * config.target_speed

    # Interceptor at origin
    interceptor_start = Vec3(0, 0, 600)
    interceptor_velocity = Vec3(config.interceptor_speed, 0, 20)  # Initial heading east

    # Create guidance function
    guidance_params = GuidanceParams(nav_constant=config.nav_constant)
    guidance = create_guidance_function(config.guidance_type, guidance_params)

    # Run simulations
    results = []
    for _ in range(config.runs_per_point):
        sim_config = SimConfig(
            dt=0.02,
            max_time=config.max_time,
            kill_radius=config.kill_radius,
            real_time=False,
        )

        engine = SimEngine(
            config=sim_config,
            guidance=guidance,
            evasion_type=config.evasion_type,
        )

        engine.setup_scenario(
            target_start=target_start,
            target_velocity=target_velocity,
            interceptor_start=interceptor_start,
            interceptor_velocity=interceptor_velocity,
        )

        final_state = await engine.run()
        results.append(final_state)

    # Compute statistics
    intercepts = [r for r in results if r.result == EngagementResult.INTERCEPT]
    intercept_rate = len(intercepts) / len(results) if results else 0

    miss_distances = [r.miss_distance for r in results]
    mean_miss = np.mean(miss_distances) if miss_distances else 0

    times = [r.sim_time for r in intercepts]
    mean_time = np.mean(times) if times else 0

    return EnvelopePoint(
        range_m=range_m,
        bearing_deg=bearing_deg,
        elevation_deg=elevation_deg,
        intercept_rate=intercept_rate,
        mean_miss_distance=mean_miss,
        mean_time_to_intercept=mean_time,
        num_runs=len(results),
    )


async def compute_engagement_envelope(config: EnvelopeConfig) -> EnvelopeResults:
    """
    Compute the full engagement envelope by sweeping parameters.

    Returns a grid of intercept probabilities across range, bearing, and elevation.
    """
    # Generate sweep values
    range_values = np.linspace(
        config.range_min, config.range_max, config.range_steps
    ).tolist()
    bearing_values = np.linspace(
        config.bearing_min, config.bearing_max, config.bearing_steps
    ).tolist()
    elevation_values = np.linspace(
        config.elevation_min, config.elevation_max, config.elevation_steps
    ).tolist()

    # Run all points concurrently
    tasks = []
    for r in range_values:
        for b in bearing_values:
            for e in elevation_values:
                tasks.append(run_envelope_point(config, r, b, e))

    points = await asyncio.gather(*tasks)

    return EnvelopeResults(
        config={
            "range_min": config.range_min,
            "range_max": config.range_max,
            "range_steps": config.range_steps,
            "bearing_min": config.bearing_min,
            "bearing_max": config.bearing_max,
            "bearing_steps": config.bearing_steps,
            "elevation_min": config.elevation_min,
            "elevation_max": config.elevation_max,
            "elevation_steps": config.elevation_steps,
            "runs_per_point": config.runs_per_point,
            "guidance_type": config.guidance_type.value,
            "nav_constant": config.nav_constant,
            "evasion_type": config.evasion_type.value,
        },
        points=list(points),
        range_values=range_values,
        bearing_values=bearing_values,
        elevation_values=elevation_values,
    )
