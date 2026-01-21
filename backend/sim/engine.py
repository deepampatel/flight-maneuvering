"""
Simulation Engine - The Heart of Real-Time Simulation

KEY CONCEPT: Fixed Timestep Simulation

Why fixed timestep (not variable)?
1. DETERMINISM: Same inputs = same outputs (crucial for replay/debugging)
2. STABILITY: Physics integration can explode with large dt
3. FAIRNESS: Behavior doesn't change based on computer speed

How it works:
- Sim runs at fixed rate (e.g., 50 Hz = dt of 0.02 seconds)
- Each "tick" advances simulation time by exactly dt
- Real time and sim time may diverge (sim can run faster/slower)

For real-time visualization, we try to match wall-clock time,
but the sim itself doesn't care - it just processes ticks.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional
import uuid

from .vector import Vec3
from .entities import Entity, EntityType, create_target, create_interceptor


class SimStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class EngagementResult(str, Enum):
    PENDING = "pending"
    INTERCEPT = "intercept"  # Success! Hit the target
    TIMEOUT = "timeout"      # Ran out of time
    MISSED = "missed"        # Passed target without hitting


@dataclass
class SimConfig:
    """Configuration for a simulation run."""
    dt: float = 0.02            # Timestep (50 Hz)
    max_time: float = 60.0      # Max sim duration (seconds)
    kill_radius: float = 150.0  # Intercept success radius (meters) - proximity fuse
    real_time: bool = True      # Try to match wall-clock time


@dataclass
class SimState:
    """Complete state of the simulation at a point in time."""
    run_id: str
    sim_time: float
    tick: int
    status: SimStatus
    result: EngagementResult
    target: Entity
    interceptor: Entity
    miss_distance: float = float('inf')

    def to_event(self) -> dict:
        """Convert to event for transmission."""
        return {
            "type": "state",
            "run_id": self.run_id,
            "ts": time.time(),
            "sim_time": self.sim_time,
            "tick": self.tick,
            "status": self.status.value,
            "result": self.result.value,
            "entities": [
                self.target.to_state_dict(),
                self.interceptor.to_state_dict(),
            ],
            "miss_distance": self.miss_distance,
        }


# Type for guidance function: takes state, returns acceleration command
GuidanceFunction = Callable[[SimState], Vec3]


def pure_pursuit_guidance(state: SimState) -> Vec3:
    """
    PURE PURSUIT: The simplest guidance law.

    Strategy: Point directly at the target's CURRENT position.

    Pros: Simple, intuitive
    Cons: Inefficient path, can't hit maneuvering targets well

    The math:
    1. Get vector from interceptor to target
    2. Normalize it (unit vector pointing at target)
    3. Scale by desired acceleration magnitude

    This is like a dog chasing a rabbit - always running
    directly toward where the rabbit IS, not where it WILL BE.
    """
    # Vector from interceptor to target
    to_target = state.target.position - state.interceptor.position
    distance = to_target.magnitude()

    if distance < 1.0:  # Very close, reduce acceleration
        return Vec3.zero()

    # Direction to target
    direction = to_target.normalized()

    # Command max acceleration toward target
    accel_magnitude = state.interceptor.max_accel

    return direction * accel_magnitude


class SimEngine:
    """
    The simulation engine.

    Responsibilities:
    1. Maintain simulation state
    2. Run the fixed-timestep loop
    3. Apply guidance laws
    4. Detect end conditions
    5. Emit events for UI/logging
    """

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        guidance: Optional[GuidanceFunction] = None,
    ):
        self.config = config or SimConfig()
        self.guidance = guidance or pure_pursuit_guidance
        self.state: Optional[SimState] = None
        self._event_handlers: list[Callable[[dict], Awaitable[None]]] = []

    def on_event(self, handler: Callable[[dict], Awaitable[None]]) -> None:
        """Register an event handler (for WebSocket broadcast, logging, etc)."""
        self._event_handlers.append(handler)

    async def _emit_event(self, event: dict) -> None:
        """Send event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")

    def setup_scenario(
        self,
        target_start: Vec3,
        target_velocity: Vec3,
        interceptor_start: Vec3,
        interceptor_velocity: Vec3,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a new scenario.

        Typical setup:
        - Target flying across the field
        - Interceptor starting from a base/launch point
        """
        self.state = SimState(
            run_id=run_id or str(uuid.uuid4())[:8],
            sim_time=0.0,
            tick=0,
            status=SimStatus.READY,
            result=EngagementResult.PENDING,
            target=create_target(target_start, target_velocity),
            interceptor=create_interceptor(interceptor_start, interceptor_velocity),
        )

    def _check_end_conditions(self) -> None:
        """Check if simulation should end."""
        if self.state is None:
            return

        # Calculate current miss distance
        self.state.miss_distance = self.state.interceptor.position.distance_to(
            self.state.target.position
        )

        # Check intercept (hit)
        if self.state.miss_distance <= self.config.kill_radius:
            self.state.status = SimStatus.COMPLETED
            self.state.result = EngagementResult.INTERCEPT
            return

        # Check timeout
        if self.state.sim_time >= self.config.max_time:
            self.state.status = SimStatus.COMPLETED
            self.state.result = EngagementResult.TIMEOUT
            return

        # Check if interceptor passed target (opening fast means we missed)
        # Only check after initial approach phase (first 2 seconds)
        if self.state.sim_time > 2.0:
            to_target = self.state.target.position - self.state.interceptor.position
            # Positive = closing, Negative = opening
            closing_velocity = to_target.normalized().dot(
                self.state.interceptor.velocity - self.state.target.velocity
            )
            if closing_velocity < -50 and self.state.miss_distance > 100:
                # Moving away fast and far - missed
                self.state.status = SimStatus.COMPLETED
                self.state.result = EngagementResult.MISSED

    async def tick(self) -> None:
        """
        Execute one simulation step.

        This is the core loop that runs at 50 Hz:
        1. Apply guidance law to get acceleration command
        2. Update entity physics
        3. Check end conditions
        4. Emit state event
        """
        if self.state is None or self.state.status != SimStatus.RUNNING:
            return

        dt = self.config.dt

        # 1. GUIDANCE: Calculate interceptor acceleration
        accel_cmd = self.guidance(self.state)
        self.state.interceptor.set_acceleration(accel_cmd)

        # 2. PHYSICS: Update all entities
        self.state.target.update(dt)
        self.state.interceptor.update(dt)

        # 3. Update sim time
        self.state.sim_time += dt
        self.state.tick += 1

        # 4. Check end conditions
        self._check_end_conditions()

        # 5. Emit state event
        await self._emit_event(self.state.to_event())

    async def run(self) -> SimState:
        """
        Run the simulation to completion.

        This manages real-time pacing if configured.
        """
        if self.state is None:
            raise RuntimeError("No scenario set up. Call setup_scenario first.")

        self.state.status = SimStatus.RUNNING

        # Emit initial state
        await self._emit_event(self.state.to_event())

        wall_start = time.monotonic()
        sim_start = self.state.sim_time

        while self.state.status == SimStatus.RUNNING:
            tick_start = time.monotonic()

            # Run one simulation step
            await self.tick()

            # Real-time pacing
            if self.config.real_time and self.state.status == SimStatus.RUNNING:
                # How much wall time SHOULD have passed?
                expected_wall = (self.state.sim_time - sim_start)
                actual_wall = time.monotonic() - wall_start
                sleep_time = expected_wall - actual_wall

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        # Emit final event
        await self._emit_event({
            "type": "complete",
            "run_id": self.state.run_id,
            "ts": time.time(),
            "result": self.state.result.value,
            "final_miss_distance": self.state.miss_distance,
            "sim_time": self.state.sim_time,
            "ticks": self.state.tick,
        })

        return self.state


# Pre-built scenarios for easy testing
# Note: Pure pursuit works best against slow/straight targets
# Phase 2 will add proportional navigation for better performance
SCENARIOS = {
    "head_on": {
        "description": "Classic head-on intercept",
        "target_start": Vec3(3000, 0, 800),       # 3km east, 800m up
        "target_velocity": Vec3(-100, 0, 0),      # Flying west at 100 m/s (slower)
        "interceptor_start": Vec3(0, 0, 600),     # At origin, 600m up
        "interceptor_velocity": Vec3(150, 0, 20), # Initial eastward, slight climb
    },
    "crossing": {
        "description": "Target crossing perpendicular to interceptor",
        "target_start": Vec3(2000, -1500, 800),
        "target_velocity": Vec3(0, 80, 0),        # Flying north at 80 m/s
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(100, 50, 15),
    },
    "tail_chase": {
        "description": "Interceptor chasing from behind",
        "target_start": Vec3(800, 0, 700),
        "target_velocity": Vec3(80, 0, 0),        # Flying away at 80 m/s
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(200, 0, 10), # Much faster than target
    },
}


def load_scenario(name: str) -> dict:
    """Get scenario parameters by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
