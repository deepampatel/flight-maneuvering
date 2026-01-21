"""
Monte Carlo Analysis - Testing Under Uncertainty

Real-world systems face uncertainty:
- Sensor noise
- Initial position errors
- Target behavior variations
- Environmental factors

Monte Carlo simulation runs MANY scenarios with random variations
to understand how robust our guidance is.

KEY CONCEPTS:

1. PARAMETER SWEEPS
   Run same scenario with varying parameters to find optimal settings.
   Example: What navigation constant N gives best results?

2. UNCERTAINTY ANALYSIS
   Add noise to initial conditions to test robustness.
   Example: How well does PN work with 100m position uncertainty?

3. ENGAGEMENT ENVELOPES
   Find the region where intercepts succeed.
   Example: What launch ranges work? What angles?
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .vector import Vec3
from .engine import SimEngine, SimConfig, SimState, EngagementResult
from .guidance import (
    GuidanceType,
    GuidanceParams,
    create_guidance_function,
)


@dataclass
class MonteCarloConfig:
    """Configuration for a Monte Carlo batch run."""
    # Base scenario
    target_start: Vec3
    target_velocity: Vec3
    interceptor_start: Vec3
    interceptor_velocity: Vec3

    # Guidance settings
    guidance_type: GuidanceType = GuidanceType.PROPORTIONAL_NAV
    nav_constant: float = 4.0

    # Simulation settings
    dt: float = 0.02
    max_time: float = 60.0
    kill_radius: float = 50.0  # Tighter for Monte Carlo analysis

    # Monte Carlo settings
    num_runs: int = 100

    # Uncertainty (standard deviations for Gaussian noise)
    position_noise_std: float = 0.0  # meters
    velocity_noise_std: float = 0.0  # m/s
    target_heading_noise_std: float = 0.0  # degrees


@dataclass
class RunResult:
    """Result from a single simulation run."""
    run_id: str
    result: str  # 'intercept', 'missed', 'timeout'
    miss_distance: float
    time_to_intercept: float
    ticks: int


@dataclass
class MonteCarloResults:
    """Aggregated results from a Monte Carlo batch."""
    config: Dict[str, Any]
    num_runs: int
    results: List[RunResult]

    # Computed statistics
    intercept_rate: float = 0.0
    miss_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_miss_distance: float = 0.0
    std_miss_distance: float = 0.0
    min_miss_distance: float = 0.0
    max_miss_distance: float = 0.0
    mean_time_to_intercept: float = 0.0

    def compute_stats(self):
        """Calculate statistics from individual results."""
        if not self.results:
            return

        intercepts = [r for r in self.results if r.result == 'intercept']
        misses = [r for r in self.results if r.result == 'missed']
        timeouts = [r for r in self.results if r.result == 'timeout']

        self.intercept_rate = len(intercepts) / len(self.results)
        self.miss_rate = len(misses) / len(self.results)
        self.timeout_rate = len(timeouts) / len(self.results)

        miss_distances = [r.miss_distance for r in self.results]
        self.mean_miss_distance = np.mean(miss_distances)
        self.std_miss_distance = np.std(miss_distances)
        self.min_miss_distance = np.min(miss_distances)
        self.max_miss_distance = np.max(miss_distances)

        if intercepts:
            self.mean_time_to_intercept = np.mean([r.time_to_intercept for r in intercepts])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON response."""
        return {
            "config": self.config,
            "num_runs": self.num_runs,
            "intercept_rate": self.intercept_rate,
            "miss_rate": self.miss_rate,
            "timeout_rate": self.timeout_rate,
            "mean_miss_distance": self.mean_miss_distance,
            "std_miss_distance": self.std_miss_distance,
            "min_miss_distance": self.min_miss_distance,
            "max_miss_distance": self.max_miss_distance,
            "mean_time_to_intercept": self.mean_time_to_intercept,
            "miss_distance_histogram": self._compute_histogram(),
        }

    def _compute_histogram(self, bins: int = 20) -> Dict[str, List]:
        """Compute histogram of miss distances for visualization."""
        if not self.results:
            return {"bins": [], "counts": []}

        miss_distances = [r.miss_distance for r in self.results]
        counts, bin_edges = np.histogram(miss_distances, bins=bins)

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }


def add_noise(vec: Vec3, std: float) -> Vec3:
    """Add Gaussian noise to a vector."""
    if std <= 0:
        return vec
    return Vec3(
        vec.x + random.gauss(0, std),
        vec.y + random.gauss(0, std),
        vec.z + random.gauss(0, std),
    )


def rotate_velocity(vel: Vec3, heading_noise_deg: float) -> Vec3:
    """Add heading noise by rotating velocity vector in horizontal plane."""
    if heading_noise_deg <= 0:
        return vel

    angle = random.gauss(0, heading_noise_deg) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Rotate in XY plane (horizontal)
    return Vec3(
        vel.x * cos_a - vel.y * sin_a,
        vel.x * sin_a + vel.y * cos_a,
        vel.z,
    )


async def run_single_sim(
    config: MonteCarloConfig,
    run_number: int,
) -> RunResult:
    """
    Run a single simulation with noise applied.

    This is called many times for Monte Carlo analysis.
    """
    # Apply noise to initial conditions
    target_start = add_noise(config.target_start, config.position_noise_std)
    target_vel = rotate_velocity(
        add_noise(config.target_velocity, config.velocity_noise_std),
        config.target_heading_noise_std
    )
    interceptor_start = add_noise(config.interceptor_start, config.position_noise_std)
    interceptor_vel = add_noise(config.interceptor_velocity, config.velocity_noise_std)

    # Create guidance function
    guidance_params = GuidanceParams(nav_constant=config.nav_constant)
    guidance = create_guidance_function(config.guidance_type, guidance_params)

    # Create and run simulation
    sim_config = SimConfig(
        dt=config.dt,
        max_time=config.max_time,
        kill_radius=config.kill_radius,
        real_time=False,  # Fast mode for Monte Carlo
    )
    engine = SimEngine(config=sim_config, guidance=guidance)
    engine.setup_scenario(
        target_start=target_start,
        target_velocity=target_vel,
        interceptor_start=interceptor_start,
        interceptor_velocity=interceptor_vel,
        run_id=f"mc_{run_number:04d}",
    )

    # Run simulation
    final_state = await engine.run()

    return RunResult(
        run_id=final_state.run_id,
        result=final_state.result.value,
        miss_distance=final_state.miss_distance,
        time_to_intercept=final_state.sim_time,
        ticks=final_state.tick,
    )


async def run_monte_carlo(config: MonteCarloConfig) -> MonteCarloResults:
    """
    Run a full Monte Carlo batch.

    Executes many simulations with random variations and aggregates results.
    """
    # Run all simulations
    tasks = [
        run_single_sim(config, i)
        for i in range(config.num_runs)
    ]
    results = await asyncio.gather(*tasks)

    # Create results object
    mc_results = MonteCarloResults(
        config={
            "guidance_type": config.guidance_type.value,
            "nav_constant": config.nav_constant,
            "kill_radius": config.kill_radius,
            "num_runs": config.num_runs,
            "position_noise_std": config.position_noise_std,
            "velocity_noise_std": config.velocity_noise_std,
        },
        num_runs=config.num_runs,
        results=list(results),
    )
    mc_results.compute_stats()

    return mc_results


async def parameter_sweep(
    base_config: MonteCarloConfig,
    param_name: str,
    param_values: List[float],
) -> List[MonteCarloResults]:
    """
    Sweep a parameter across multiple values.

    Example: Sweep nav_constant from 2.0 to 6.0 to find optimal value.
    """
    results = []

    for value in param_values:
        # Clone config and update parameter
        config = MonteCarloConfig(
            target_start=base_config.target_start,
            target_velocity=base_config.target_velocity,
            interceptor_start=base_config.interceptor_start,
            interceptor_velocity=base_config.interceptor_velocity,
            guidance_type=base_config.guidance_type,
            nav_constant=base_config.nav_constant,
            dt=base_config.dt,
            max_time=base_config.max_time,
            kill_radius=base_config.kill_radius,
            num_runs=base_config.num_runs,
            position_noise_std=base_config.position_noise_std,
            velocity_noise_std=base_config.velocity_noise_std,
            target_heading_noise_std=base_config.target_heading_noise_std,
        )

        # Update the swept parameter
        if param_name == "nav_constant":
            config.nav_constant = value
        elif param_name == "kill_radius":
            config.kill_radius = value
        elif param_name == "position_noise_std":
            config.position_noise_std = value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Run Monte Carlo for this parameter value
        mc_result = await run_monte_carlo(config)
        mc_result.config["swept_param"] = param_name
        mc_result.config["swept_value"] = value
        results.append(mc_result)

    return results
