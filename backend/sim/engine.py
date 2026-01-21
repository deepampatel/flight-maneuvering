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
from typing import Callable, Awaitable, Optional, List, Dict
import uuid

from .vector import Vec3
from .entities import Entity, EntityType, create_target, create_interceptor
from .evasion import (
    EvasionType, EvasionConfig, EvasionState,
    EvasionFunction, create_evasion_function, no_evasion
)
from .assignment import (
    WTAAlgorithm, AssignmentResult, compute_assignment
)


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
    targets: List[Entity]  # Support multiple targets (Phase 5)
    interceptors: List[Entity]  # Support multiple interceptors
    miss_distance: float = float('inf')
    intercepting_id: Optional[str] = None  # Which interceptor hit (if any)
    intercepted_target_id: Optional[str] = None  # Which target was hit (Phase 5)
    intercepted_pairs: List[tuple] = field(default_factory=list)  # All (interceptor_id, target_id) pairs that hit

    # Legacy property for backward compatibility - single target
    @property
    def target(self) -> Entity:
        """Get first target (for backward compatibility)."""
        return self.targets[0] if self.targets else None

    # Legacy property for backward compatibility - single interceptor
    @property
    def interceptor(self) -> Entity:
        """Get first interceptor (for backward compatibility)."""
        return self.interceptors[0] if self.interceptors else None

    def to_event(self, assignments: Optional['AssignmentResult'] = None) -> dict:
        """Convert to event for transmission."""
        # Include all targets and interceptors in entities list
        entities = [t.to_state_dict() for t in self.targets]
        entities.extend(i.to_state_dict() for i in self.interceptors)

        event = {
            "type": "state",
            "run_id": self.run_id,
            "ts": time.time(),
            "sim_time": self.sim_time,
            "tick": self.tick,
            "status": self.status.value,
            "result": self.result.value,
            "entities": entities,
            "miss_distance": self.miss_distance,
            "intercepting_id": self.intercepting_id,
            "intercepted_target_id": self.intercepted_target_id,
            "intercepted_pairs": self.intercepted_pairs,
        }

        # Include WTA assignments if available
        if assignments:
            event["assignments"] = assignments.to_dict()

        return event


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
    4. Apply target evasion maneuvers
    5. Detect end conditions
    6. Emit events for UI/logging
    """

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        guidance: Optional[GuidanceFunction] = None,
        evasion_type: EvasionType = EvasionType.NONE,
        evasion_config: Optional[EvasionConfig] = None,
        wta_algorithm: WTAAlgorithm = WTAAlgorithm.HUNGARIAN,
    ):
        self.config = config or SimConfig()
        self.guidance = guidance or pure_pursuit_guidance
        self.state: Optional[SimState] = None
        self._event_handlers: list[Callable[[dict], Awaitable[None]]] = []

        # Evasion setup
        self.evasion_type = evasion_type
        evasion_fn, evasion_state, evasion_cfg = create_evasion_function(
            evasion_type, evasion_config
        )
        self._evasion_fn = evasion_fn
        self._evasion_state = evasion_state
        self._evasion_config = evasion_cfg

        # Phase 5: WTA setup
        self.wta_algorithm = wta_algorithm
        self.assignments: Optional[AssignmentResult] = None

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
        num_interceptors: int = 1,
        interceptor_spacing: float = 200.0,  # meters between interceptors
        # Phase 5: Multi-target support
        additional_targets: Optional[List[Dict]] = None,  # [{start: Vec3, velocity: Vec3}, ...]
        num_targets: int = 1,  # Alternative: auto-generate multiple targets
        target_spacing: float = 300.0,  # meters between auto-generated targets
    ) -> None:
        """
        Initialize a new scenario.

        Typical setup:
        - One or more targets flying across the field
        - One or more interceptors starting from a base/launch point

        Args:
            num_interceptors: Number of interceptors to spawn
            interceptor_spacing: Lateral spacing between interceptors (meters)
            additional_targets: List of additional target configs [{start: Vec3, velocity: Vec3}, ...]
            num_targets: Auto-generate this many targets (spread laterally)
            target_spacing: Lateral spacing for auto-generated targets
        """
        # Create interceptors with lateral offset
        interceptors = []
        for i in range(num_interceptors):
            # Offset perpendicular to initial velocity direction
            if num_interceptors > 1:
                # Center the formation around the base position
                offset_index = i - (num_interceptors - 1) / 2
                # Perpendicular direction in XY plane
                vel_mag = interceptor_velocity.magnitude()
                if vel_mag > 0:
                    perp = Vec3(
                        -interceptor_velocity.y / vel_mag,
                        interceptor_velocity.x / vel_mag,
                        0
                    )
                else:
                    perp = Vec3(0, 1, 0)
                offset = perp * (offset_index * interceptor_spacing)
                start_pos = interceptor_start + offset
            else:
                start_pos = interceptor_start

            interceptors.append(
                create_interceptor(start_pos, interceptor_velocity, f"I{i+1}")
            )

        # Phase 5: Create targets (support multiple)
        targets = []

        # Primary target always created
        if num_targets == 1 and not additional_targets:
            # Single target - backward compatible
            targets.append(create_target(target_start, target_velocity, "T1"))
        else:
            # Multiple targets - either from num_targets or additional_targets
            if additional_targets:
                # Use explicit target definitions
                targets.append(create_target(target_start, target_velocity, "T1"))
                for i, target_cfg in enumerate(additional_targets):
                    targets.append(create_target(
                        target_cfg["start"],
                        target_cfg["velocity"],
                        f"T{i+2}"
                    ))
            else:
                # Auto-generate targets with lateral spacing
                for i in range(num_targets):
                    if num_targets > 1:
                        # Offset perpendicular to target velocity
                        offset_index = i - (num_targets - 1) / 2
                        vel_mag = target_velocity.magnitude()
                        if vel_mag > 0:
                            perp = Vec3(
                                -target_velocity.y / vel_mag,
                                target_velocity.x / vel_mag,
                                0
                            )
                        else:
                            perp = Vec3(0, 1, 0)
                        offset = perp * (offset_index * target_spacing)
                        start_pos = target_start + offset
                    else:
                        start_pos = target_start

                    targets.append(create_target(start_pos, target_velocity, f"T{i+1}"))

        # Reset evasion state for new scenario
        self._evasion_state = EvasionState()
        # Create evasion states per target for multi-target scenarios
        self._evasion_states: Dict[str, EvasionState] = {
            t.id: EvasionState() for t in targets
        }

        self.state = SimState(
            run_id=run_id or str(uuid.uuid4())[:8],
            sim_time=0.0,
            tick=0,
            status=SimStatus.READY,
            result=EngagementResult.PENDING,
            targets=targets,
            interceptors=interceptors,
        )

        # Phase 5: Compute initial WTA assignment (once at start)
        if len(targets) > 0 and len(interceptors) > 0:
            self.assignments = compute_assignment(
                interceptors,
                targets,
                self.wta_algorithm
            )
        else:
            self.assignments = None

    def _check_end_conditions(self) -> None:
        """Check if simulation should end.

        Phase 5: Updated to handle multiple targets.
        Intercept is achieved when ALL targets are hit (or all interceptors/targets are accounted for).
        """
        if self.state is None:
            return

        # Track which targets have been intercepted this tick
        active_targets = [t for t in self.state.targets if t.id not in
                         [pair[1] for pair in self.state.intercepted_pairs]]
        active_interceptors = [i for i in self.state.interceptors if i.id not in
                              [pair[0] for pair in self.state.intercepted_pairs]]

        # Calculate miss distance (minimum across all active interceptor-target pairs)
        min_miss_distance = float('inf')

        for interceptor in active_interceptors:
            for target in active_targets:
                dist = interceptor.position.distance_to(target.position)
                if dist < min_miss_distance:
                    min_miss_distance = dist

        self.state.miss_distance = min_miss_distance

        # Check intercept (any interceptor hits any target)
        new_intercepts = []
        for interceptor in active_interceptors:
            for target in active_targets:
                dist = interceptor.position.distance_to(target.position)
                if dist <= self.config.kill_radius:
                    new_intercepts.append((interceptor.id, target.id))
                    # Record the first intercept details for backward compatibility
                    if self.state.intercepting_id is None:
                        self.state.intercepting_id = interceptor.id
                        self.state.intercepted_target_id = target.id

        # Add new intercepts to the list
        for pair in new_intercepts:
            if pair not in self.state.intercepted_pairs:
                self.state.intercepted_pairs.append(pair)

        # Check if all targets have been intercepted
        intercepted_target_ids = set(pair[1] for pair in self.state.intercepted_pairs)
        all_target_ids = set(t.id for t in self.state.targets)

        if intercepted_target_ids == all_target_ids:
            # All targets hit - success!
            self.state.status = SimStatus.COMPLETED
            self.state.result = EngagementResult.INTERCEPT
            return

        # Check timeout
        if self.state.sim_time >= self.config.max_time:
            self.state.status = SimStatus.COMPLETED
            self.state.result = EngagementResult.TIMEOUT
            return

        # Check if all active interceptors have missed all active targets
        # Only check after initial approach phase (first 2 seconds)
        if self.state.sim_time > 2.0 and active_interceptors and active_targets:
            all_missed = True
            for interceptor in active_interceptors:
                for target in active_targets:
                    to_target = target.position - interceptor.position
                    dist = to_target.magnitude()
                    if dist < 1.0:
                        all_missed = False
                        continue
                    # Positive = closing, Negative = opening
                    closing_velocity = to_target.normalized().dot(
                        interceptor.velocity - target.velocity
                    )
                    # Still approaching or close enough
                    if closing_velocity > -50 or dist < 100:
                        all_missed = False
                        break
                if not all_missed:
                    break

            if all_missed:
                self.state.status = SimStatus.COMPLETED
                self.state.result = EngagementResult.MISSED

    async def tick(self) -> None:
        """
        Execute one simulation step.

        This is the core loop that runs at 50 Hz:
        1. Apply target evasion maneuver (for each target)
        2. Apply guidance law to each interceptor (against assigned/nearest target)
        3. Update entity physics
        4. Check end conditions
        5. Emit state event
        """
        if self.state is None or self.state.status != SimStatus.RUNNING:
            return

        dt = self.config.dt

        # Track which targets are still active (not yet intercepted)
        intercepted_target_ids = set(pair[1] for pair in self.state.intercepted_pairs)
        active_targets = [t for t in self.state.targets if t.id not in intercepted_target_ids]

        # 1. EVASION: Calculate evasion for each active target
        for target in active_targets:
            # Get or create evasion state for this target
            evasion_state = self._evasion_states.get(target.id, self._evasion_state)
            evasion_accel = self._evasion_fn(
                target.position,
                target.velocity,
                dt,
                evasion_state,
                self._evasion_config,
            )
            target.set_acceleration(evasion_accel)

        # 2. GUIDANCE: Calculate acceleration for each interceptor
        # Track which interceptors are still active
        intercepted_interceptor_ids = set(pair[0] for pair in self.state.intercepted_pairs)

        for interceptor in self.state.interceptors:
            # Skip interceptors that have already hit a target
            if interceptor.id in intercepted_interceptor_ids:
                interceptor.set_acceleration(Vec3.zero())
                # Also zero velocity so they stop moving after hit
                interceptor.velocity = Vec3.zero()
                continue

            if not active_targets:
                # No targets left - coast
                interceptor.set_acceleration(Vec3.zero())
                continue

            # Phase 5: Use WTA assignment to get assigned target
            assigned_target = None
            if self.assignments:
                assigned_target_id = self.assignments.get_target_for_interceptor(interceptor.id)
                if assigned_target_id:
                    # Find the target entity - but only if it's still active
                    for t in active_targets:
                        if t.id == assigned_target_id:
                            assigned_target = t
                            break

            # Fallback: if no assignment or assigned target already hit, use nearest
            if assigned_target is None:
                assigned_target = min(
                    active_targets,
                    key=lambda t: interceptor.position.distance_to(t.position)
                )

            # Create a temporary state view for this interceptor-target pair
            # This allows guidance to work with single interceptor/target abstraction
            temp_state = SimState(
                run_id=self.state.run_id,
                sim_time=self.state.sim_time,
                tick=self.state.tick,
                status=self.state.status,
                result=self.state.result,
                targets=[assigned_target],  # Single target view
                interceptors=[interceptor],  # Single interceptor view
                miss_distance=interceptor.position.distance_to(assigned_target.position),
            )
            accel_cmd = self.guidance(temp_state)
            interceptor.set_acceleration(accel_cmd)

        # 3. PHYSICS: Update all entities
        for target in self.state.targets:
            target.update(dt)
        for interceptor in self.state.interceptors:
            interceptor.update(dt)

        # 4. Update sim time
        self.state.sim_time += dt
        self.state.tick += 1

        # 5. Check end conditions
        self._check_end_conditions()

        # 6. Emit state event (include WTA assignments)
        await self._emit_event(self.state.to_event(self.assignments))

    async def run(self) -> SimState:
        """
        Run the simulation to completion.

        This manages real-time pacing if configured.
        Handles cancellation gracefully by emitting a stopped event.
        """
        if self.state is None:
            raise RuntimeError("No scenario set up. Call setup_scenario first.")

        self.state.status = SimStatus.RUNNING

        # Emit initial state (include WTA assignments)
        await self._emit_event(self.state.to_event(self.assignments))

        wall_start = time.monotonic()
        sim_start = self.state.sim_time

        try:
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

        except asyncio.CancelledError:
            # Handle graceful cancellation (e.g., user clicked ABORT)
            self.state.status = SimStatus.COMPLETED
            self.state.result = EngagementResult.TIMEOUT  # Mark as stopped/timeout

            # Shield emit operations from cancellation so UI gets the update
            try:
                # Emit final state so UI updates immediately
                await asyncio.shield(self._emit_event(self.state.to_event(self.assignments)))

                # Emit stopped event so UI knows simulation ended
                await asyncio.shield(self._emit_event({
                    "type": "complete",
                    "run_id": self.state.run_id,
                    "ts": time.time(),
                    "result": "stopped",
                    "final_miss_distance": self.state.miss_distance,
                    "sim_time": self.state.sim_time,
                    "ticks": self.state.tick,
                }))
            except asyncio.CancelledError:
                pass  # Ignore if shield still gets cancelled

            raise  # Re-raise so the caller knows it was cancelled

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
SCENARIOS = {
    "head_on": {
        "description": "Classic head-on intercept",
        "target_start": Vec3(3000, 0, 800),       # 3km east, 800m up
        "target_velocity": Vec3(-100, 0, 0),      # Flying west at 100 m/s (slower)
        "interceptor_start": Vec3(0, 0, 600),     # At origin, 600m up
        "interceptor_velocity": Vec3(150, 0, 20), # Initial eastward, slight climb
        "evasion": EvasionType.NONE,
    },
    "crossing": {
        "description": "Target crossing perpendicular to interceptor",
        "target_start": Vec3(2000, -1500, 800),
        "target_velocity": Vec3(0, 80, 0),        # Flying north at 80 m/s
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(100, 50, 15),
        "evasion": EvasionType.NONE,
    },
    "tail_chase": {
        "description": "Interceptor chasing from behind",
        "target_start": Vec3(800, 0, 700),
        "target_velocity": Vec3(80, 0, 0),        # Flying away at 80 m/s
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(200, 0, 10), # Much faster than target
        "evasion": EvasionType.NONE,
    },
    # Evasive scenarios
    "head_on_turning": {
        "description": "Head-on with constant turn evasion",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.CONSTANT_TURN,
    },
    "head_on_weaving": {
        "description": "Head-on with weave (S-turn) evasion",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.WEAVE,
    },
    "head_on_barrel_roll": {
        "description": "Head-on with 3D barrel roll evasion",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.BARREL_ROLL,
    },
    "head_on_jinking": {
        "description": "Head-on with random jink evasion",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.RANDOM_JINK,
    },
    "crossing_weaving": {
        "description": "Crossing target with weave evasion",
        "target_start": Vec3(2000, -1500, 800),
        "target_velocity": Vec3(0, 80, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(100, 50, 15),
        "evasion": EvasionType.WEAVE,
    },
    "tail_chase_jinking": {
        "description": "Tail chase with random jink evasion",
        "target_start": Vec3(800, 0, 700),
        "target_velocity": Vec3(80, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(200, 0, 10),
        "evasion": EvasionType.RANDOM_JINK,
    },
    # Phase 5: Multi-target scenarios
    "multi_target_2": {
        "description": "Two targets, head-on approach",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.NONE,
        "num_targets": 2,
        "target_spacing": 400.0,
    },
    "multi_target_3": {
        "description": "Three targets, spread formation",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.NONE,
        "num_targets": 3,
        "target_spacing": 350.0,
    },
    "multi_target_weaving": {
        "description": "Two weaving targets",
        "target_start": Vec3(3000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.WEAVE,
        "num_targets": 2,
        "target_spacing": 500.0,
    },
    "salvo_attack": {
        "description": "Four targets - saturation attack",
        "target_start": Vec3(3500, 0, 900),
        "target_velocity": Vec3(-120, 0, -10),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(180, 0, 30),
        "evasion": EvasionType.NONE,
        "num_targets": 4,
        "target_spacing": 300.0,
    },
}


def load_scenario(name: str) -> dict:
    """Get scenario parameters by name.

    Returns scenario dict with keys:
    - target_start, target_velocity: Primary target position/velocity
    - interceptor_start, interceptor_velocity: Interceptor position/velocity
    - evasion: EvasionType (optional)
    - num_targets: Number of targets to spawn (optional, default 1)
    - target_spacing: Lateral spacing for multi-target (optional)
    - additional_targets: List of explicit target configs (optional)
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
