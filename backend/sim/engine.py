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
from .entities import Entity, EntityType, create_target, create_interceptor, create_threat_from_catalog, create_interceptor_from_catalog
from .evasion import (
    EvasionType, EvasionConfig, EvasionState,
    EvasionFunction, create_evasion_function, no_evasion
)
from .assignment import (
    WTAAlgorithm, AssignmentResult, Assignment, compute_assignment
)
from .environment import EnvironmentModel, EnvironmentConfig
from .cooperation import CooperativeEngagementManager, HandoffReason
from .ipp import ProtectedArea, ImpactPrediction, predict_impact_point, check_threat_to_areas
from .battery import Battery, BatteryConfig, create_iron_dome_battery
from .waves import WaveManager, ThreatWave
from .decoy import deploy_decoys, classify_threat, DecoyConfig

# Optional subsystem imports (features disabled by default)
try:
    from .swarm import SwarmController, SwarmConfig
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    SwarmController = None
    SwarmConfig = None

try:
    from .terrain import TerrainModel, TerrainConfig
    TERRAIN_AVAILABLE = True
except ImportError:
    TERRAIN_AVAILABLE = False
    TerrainModel = None
    TerrainConfig = None

try:
    from .datalink import DatalinkModel, DatalinkConfig
    DATALINK_AVAILABLE = True
except ImportError:
    DATALINK_AVAILABLE = False
    DatalinkModel = None
    DatalinkConfig = None

try:
    from .hmt import HumanMachineTeaming, HMTConfig, PendingAction, ActionType
    HMT_AVAILABLE = True
except ImportError:
    HMT_AVAILABLE = False
    HumanMachineTeaming = None
    HMTConfig = None

try:
    from .launcher import LaunchPlatform, LauncherConfig, create_launcher
    LAUNCHER_AVAILABLE = True
except ImportError:
    LAUNCHER_AVAILABLE = False
    LaunchPlatform = None
    LauncherConfig = None


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

    # Environmental effects
    environment: Optional[EnvironmentConfig] = None  # None = no environment effects

    # Cooperative engagement
    enable_cooperative: bool = False  # Enable cooperative engagement features

    # Swarm control
    enable_swarm: bool = False
    swarm_config: Optional['SwarmConfig'] = None

    # Terrain
    enable_terrain: bool = False
    terrain_config: Optional['TerrainConfig'] = None

    # Datalink
    enable_datalink: bool = False
    datalink_config: Optional['DatalinkConfig'] = None

    # Human-Machine Teaming
    enable_hmt: bool = False
    hmt_config: Optional['HMTConfig'] = None

    # Protected areas for IPP
    protected_areas: Optional[List['ProtectedArea']] = None

    # Battery system (Phase 5)
    batteries: Optional[List['BatteryConfig']] = None

    # Wave system (Phase 6)
    waves: Optional[List['ThreatWave']] = None
    enable_decoys: bool = False


@dataclass
class SimState:
    """Complete state of the simulation at a point in time."""
    run_id: str
    sim_time: float
    tick: int
    status: SimStatus
    result: EngagementResult
    targets: List[Entity]  # Support multiple targets
    interceptors: List[Entity]  # Support multiple interceptors
    miss_distance: float = float('inf')
    intercepting_id: Optional[str] = None  # Which interceptor hit (if any)
    intercepted_target_id: Optional[str] = None  # Which target was hit
    intercepted_pairs: List[tuple] = field(default_factory=list)  # All (interceptor_id, target_id) pairs that hit

    # Environment state
    environment: Optional['EnvironmentModel'] = None

    # Cooperative engagement state
    cooperative: Optional['CooperativeEngagementManager'] = None

    # Swarm controller
    swarm: Optional['SwarmController'] = None

    # Terrain model
    terrain: Optional['TerrainModel'] = None

    # Datalink model
    datalink: Optional['DatalinkModel'] = None

    # Human-Machine Teaming
    hmt: Optional['HumanMachineTeaming'] = None

    # Launch platforms (bogeys)
    launchers: List['LaunchPlatform'] = field(default_factory=list)

    # Protected areas and impact predictions (Phase 4)
    protected_areas: List['ProtectedArea'] = field(default_factory=list)
    impact_predictions: Dict[str, 'ImpactPrediction'] = field(default_factory=dict)

    # Battery system (Phase 5)
    batteries: List['Battery'] = field(default_factory=list)

    # Wave system (Phase 6)
    wave_manager: Optional['WaveManager'] = None
    decoy_tracking: Dict[str, float] = field(default_factory=dict)  # entity_id -> first_seen_time
    engagement_stats: Dict[str, int] = field(default_factory=lambda: {
        "total_threats": 0,
        "threats_detected": 0,
        "threats_engaged": 0,
        "threats_intercepted": 0,
        "threats_impacted": 0,
        "decoys_deployed": 0,
        "decoys_classified": 0,
        "ammo_expended": 0,
        "waves_spawned": 0,
    })

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

        # Handle infinite miss_distance (not JSON compliant)
        miss_dist = self.miss_distance if self.miss_distance != float('inf') else -1

        event = {
            "type": "state",
            "run_id": self.run_id,
            "ts": time.time(),
            "sim_time": self.sim_time,
            "tick": self.tick,
            "status": self.status.value,
            "result": self.result.value,
            "entities": entities,
            "miss_distance": miss_dist,
            "intercepting_id": self.intercepting_id,
            "intercepted_target_id": self.intercepted_target_id,
            "intercepted_pairs": self.intercepted_pairs,
        }

        # Include WTA assignments if available
        if assignments:
            event["assignments"] = assignments.to_dict()

        # Include environment state
        if self.environment:
            event["environment"] = {
                "config": self.environment.config.to_dict(),
                "current_wind": self.environment.state.current_wind.to_dict(),
            }

        # Include cooperative state
        if self.cooperative:
            event["cooperative"] = self.cooperative.get_state().to_dict()

        # Include swarm state
        if self.swarm:
            event["swarm"] = {
                "leader_id": self.swarm.state.leader_id,
                "formation_error": self.swarm.state.formation_error,
                "cohesion_metric": self.swarm.state.cohesion_metric,
                "formation": self.swarm.config.formation.value,
                "slot_positions": {
                    str(k): v.to_dict() for k, v in self.swarm.state.slot_positions.items()
                },
            }

        # Include datalink state
        if self.datalink:
            event["datalink"] = self.datalink.get_stats().to_dict()

        # Include HMT state
        if self.hmt:
            event["hmt"] = {
                "authority_level": self.hmt.config.authority_level.value,
                "pending_count": len(self.hmt.pending_actions),
                "pending_actions": [a.to_dict() for a in self.hmt.get_pending_actions()],
                "workload": self.hmt.workload.to_dict(),
                "trust": self.hmt.trust.to_dict(),
            }

        # Include launcher state
        if self.launchers:
            event["launchers"] = [l.to_state_dict() for l in self.launchers]

        # Include protected areas
        if self.protected_areas:
            event["protected_areas"] = [
                {
                    "id": area.id,
                    "name": area.name,
                    "center": area.center.to_dict(),
                    "radius": area.radius,
                    "priority": area.priority,
                    "population": area.population,
                }
                for area in self.protected_areas
            ]

        # Include impact predictions
        if self.impact_predictions:
            event["impact_predictions"] = {
                tid: {
                    "threat_id": pred.threat_id,
                    "impact_point": pred.impact_point.to_dict(),
                    "time_to_impact": pred.time_to_impact,
                    "threatens_area": pred.threatens_area,
                    "area_name": pred.area_name,
                    "distance_to_area": pred.distance_to_area,
                    "confidence": pred.confidence,
                    "engage": pred.engage,
                }
                for tid, pred in self.impact_predictions.items()
            }

        # Include battery states
        if self.batteries:
            event["batteries"] = [b.to_state_dict() for b in self.batteries]

        # Include engagement stats (Phase 6)
        if self.engagement_stats:
            event["engagement_stats"] = self.engagement_stats

        # Include wave info
        if self.wave_manager:
            event["wave_info"] = {
                "total_threats": self.wave_manager.total_threats,
                "spawned": self.wave_manager.total_spawned,
                "remaining": self.wave_manager.total_threats - self.wave_manager.total_spawned,
                "all_spawned": self.wave_manager.all_spawned,
            }

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
        environment_config: Optional[EnvironmentConfig] = None,
        enable_cooperative: bool = False,
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

        # WTA setup
        self.wta_algorithm = wta_algorithm
        self.assignments: Optional[AssignmentResult] = None

        # Environment setup
        env_cfg = environment_config or self.config.environment
        self.environment: Optional[EnvironmentModel] = (
            EnvironmentModel(env_cfg) if env_cfg else None
        )

        # Cooperative engagement setup
        self.enable_cooperative = enable_cooperative or self.config.enable_cooperative
        self.cooperative: Optional[CooperativeEngagementManager] = (
            CooperativeEngagementManager() if self.enable_cooperative else None
        )

        # Swarm controller setup
        self.swarm: Optional['SwarmController'] = None
        if self.config.enable_swarm and SWARM_AVAILABLE:
            swarm_cfg = self.config.swarm_config or SwarmConfig()
            self.swarm = SwarmController(swarm_cfg)

        # Terrain model setup
        self.terrain: Optional['TerrainModel'] = None
        if self.config.enable_terrain and TERRAIN_AVAILABLE:
            terrain_cfg = self.config.terrain_config or TerrainConfig()
            self.terrain = TerrainModel(terrain_cfg)
            self.terrain.generate_procedural()  # Auto-generate procedural terrain

        # Datalink model setup
        self.datalink: Optional['DatalinkModel'] = None
        if self.config.enable_datalink and DATALINK_AVAILABLE:
            datalink_cfg = self.config.datalink_config or DatalinkConfig()
            self.datalink = DatalinkModel(datalink_cfg)

        # Human-Machine Teaming setup
        self.hmt: Optional['HumanMachineTeaming'] = None
        if self.config.enable_hmt and HMT_AVAILABLE:
            hmt_cfg = self.config.hmt_config or HMTConfig()
            self.hmt = HumanMachineTeaming(hmt_cfg)

        # Protected areas for IPP (Phase 4)
        self.protected_areas: List[ProtectedArea] = self.config.protected_areas or []

        # Battery system (Phase 5)
        self.battery_configs: List[BatteryConfig] = self.config.batteries or []
        self.batteries: List[Battery] = []

        # Wave system (Phase 6)
        self.wave_configs: List[ThreatWave] = self.config.waves or []
        self.wave_manager: Optional[WaveManager] = None
        self.enable_decoys: bool = self.config.enable_decoys

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
        # Multi-target support
        additional_targets: Optional[List[Dict]] = None,  # [{start: Vec3, velocity: Vec3}, ...]
        num_targets: int = 1,  # Alternative: auto-generate multiple targets
        target_spacing: float = 300.0,  # meters between auto-generated targets
        protected_areas: Optional[List['ProtectedArea']] = None,
        threat_type: Optional[str] = None,  # e.g. 'qassam', 'grad' — uses catalog
        interceptor_type: Optional[str] = None,  # e.g. 'tamir' — uses catalog
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
            protected_areas: Optional list of ProtectedArea for IPP (scenario-specific)
        """
        # Merge protected areas from parameter (scenario-specific) with engine defaults
        if protected_areas is not None:
            self.protected_areas = protected_areas

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

            if interceptor_type:
                interceptors.append(
                    create_interceptor_from_catalog(interceptor_type, start_pos, interceptor_velocity, f"I{i+1}")
                )
            else:
                interceptors.append(
                    create_interceptor(start_pos, interceptor_velocity, f"I{i+1}")
                )

        # Helper: create target using catalog or default
        def _make_target(pos: Vec3, vel: Vec3, tid: str) -> Entity:
            if threat_type:
                return create_threat_from_catalog(threat_type, pos, vel, tid)
            return create_target(pos, vel, tid)

        # Create targets (support multiple)
        targets = []

        # Primary target always created
        if num_targets == 1 and not additional_targets:
            # Single target - backward compatible
            targets.append(_make_target(target_start, target_velocity, "T1"))
        else:
            # Multiple targets - either from num_targets or additional_targets
            if additional_targets:
                # Use explicit target definitions
                targets.append(_make_target(target_start, target_velocity, "T1"))
                for i, target_cfg in enumerate(additional_targets):
                    targets.append(_make_target(
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

                    targets.append(_make_target(start_pos, target_velocity, f"T{i+1}"))

        # Reset evasion state for new scenario
        self._evasion_state = EvasionState()
        # Create evasion states per target for multi-target scenarios
        self._evasion_states: Dict[str, EvasionState] = {
            t.id: EvasionState() for t in targets
        }

        # Reset cooperative manager if enabled
        if self.cooperative:
            self.cooperative.reset()

        self.state = SimState(
            run_id=run_id or str(uuid.uuid4())[:8],
            sim_time=0.0,
            tick=0,
            status=SimStatus.READY,
            result=EngagementResult.PENDING,
            targets=targets,
            interceptors=interceptors,
            environment=self.environment,
            cooperative=self.cooperative,
            swarm=self.swarm,
            terrain=self.terrain,
            datalink=self.datalink,
            hmt=self.hmt,
        )

        # Add protected areas to state
        self.state.protected_areas = self.protected_areas

        # Initialize batteries from configs (Phase 5)
        self.batteries = []
        for i, bconfig in enumerate(self.battery_configs):
            battery = Battery(f"BAT_{i+1}", bconfig)
            self.batteries.append(battery)
        # Add batteries to state
        self.state.batteries = self.batteries

        # Initialize wave manager (Phase 6)
        if self.wave_configs:
            self.wave_manager = WaveManager(self.wave_configs)
            self.state.wave_manager = self.wave_manager

        self._finalize_scenario_setup(targets, interceptors)

    def setup_custom_scenario(
        self,
        entities: List[Dict],
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a scenario from custom entity placements (Mission Planner).

        Args:
            entities: List of entity dicts with id, type, position, velocity
            run_id: Optional run ID
        """
        interceptors = []
        targets = []

        for e in entities:
            pos = Vec3(e.position.x, e.position.y, e.position.z)
            vel = Vec3(e.velocity.x, e.velocity.y, e.velocity.z)

            if e.type == 'interceptor':
                interceptors.append(create_interceptor(pos, vel, e.id))
            elif e.type == 'target':
                targets.append(create_target(pos, vel, e.id))

        # Note: interceptors list can be empty if launchers will spawn them
        # The check for "at least one interceptor OR launcher" happens in server.py
        if not targets:
            raise ValueError("At least one target required")

        # Reset evasion state for new scenario
        self._evasion_state = EvasionState()
        self._evasion_states: Dict[str, EvasionState] = {
            t.id: EvasionState() for t in targets
        }

        # Reset cooperative manager if enabled
        if self.cooperative:
            self.cooperative.reset()

        self.state = SimState(
            run_id=run_id or str(uuid.uuid4())[:8],
            sim_time=0.0,
            tick=0,
            status=SimStatus.READY,
            result=EngagementResult.PENDING,
            targets=targets,
            interceptors=interceptors,
            environment=self.environment,
            cooperative=self.cooperative,
            swarm=self.swarm,
            terrain=self.terrain,
            datalink=self.datalink,
            hmt=self.hmt,
        )

        # Add protected areas to state
        self.state.protected_areas = self.protected_areas

        # Initialize batteries from configs (Phase 5)
        self.batteries = []
        for i, bconfig in enumerate(self.battery_configs):
            battery = Battery(f"BAT_{i+1}", bconfig)
            self.batteries.append(battery)
        # Add batteries to state
        self.state.batteries = self.batteries

        self._finalize_scenario_setup(targets, interceptors)

    def _finalize_scenario_setup(
        self,
        targets: List[Entity],
        interceptors: List[Entity],
    ) -> None:
        """Common setup finalization for both standard and custom scenarios."""
        # Compute initial WTA assignment (once at start)
        if len(targets) > 0 and len(interceptors) > 0:
            self.assignments = compute_assignment(
                interceptors,
                targets,
                self.wta_algorithm,
                cooperative_manager=self.cooperative,
            )

            # Initialize target assignments in cooperative manager
            if self.cooperative and self.assignments:
                for assignment in self.assignments.assignments:
                    self.cooperative.set_target_assignment(
                        assignment.target_id,
                        assignment.interceptor_id
                    )
        else:
            self.assignments = None

    def _check_end_conditions(self) -> None:
        """Check if simulation should end.

        Updated to handle multiple targets.
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

        # Check if launchers can still spawn interceptors
        launchers_have_missiles = any(
            l.missiles_remaining > 0 for l in self.state.launchers
        ) if self.state.launchers else False

        # Check if batteries can still launch interceptors
        batteries_have_missiles = any(
            b.missiles_remaining > 0 for b in self.batteries
        ) if self.batteries else False

        can_still_launch = launchers_have_missiles or batteries_have_missiles

        # Check if all active interceptors have missed all active targets
        # Only check after initial approach phase (first 2 seconds)
        # Skip this check if launchers/batteries can still spawn more interceptors
        if self.state.sim_time > 2.0 and active_interceptors and active_targets and not can_still_launch:
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
        3. Apply environmental forces (wind, drag)
        4. Update entity physics
        5. Check end conditions
        6. Emit state event
        """
        if self.state is None or self.state.status != SimStatus.RUNNING:
            return

        dt = self.config.dt

        # 0. WAVES: Spawn new threats from scheduled waves (Phase 6)
        if self.wave_manager:
            new_wave_threats = self.wave_manager.update(self.state.sim_time)
            for threat in new_wave_threats:
                self.state.targets.append(threat)
                # Create evasion state for newly spawned threats
                self._evasion_states[threat.id] = EvasionState()
                self.state.engagement_stats["waves_spawned"] = self.wave_manager.total_spawned
                self.state.engagement_stats["total_threats"] = len(self.state.targets)

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

        # 1.5 IPP: Impact Point Prediction for all threats when areas are defended
        #     Predicts where each threat will land and whether it threatens a protected area.
        #     This is the KEY innovation of Iron Dome: only intercept what matters.
        if self.protected_areas and active_targets:
            for target in active_targets:
                prediction = check_threat_to_areas(
                    threat_id=target.id,
                    position=target.position,
                    velocity=target.velocity,
                    dry_mass=target.dry_mass,
                    cross_section=target.cross_section,
                    drag_coefficient=target.drag_coefficient,
                    propulsion=target.propulsion,
                    burn_elapsed=target.burn_elapsed,
                    guided=target.guided,
                    protected_areas=self.protected_areas,
                )
                self.state.impact_predictions[target.id] = prediction
                # Set engagement decision on the entity
                if prediction.engage:
                    target.engagement_decision = "engage"
                elif prediction.threatens_area:
                    target.engagement_decision = "track_only"
                else:
                    target.engagement_decision = "ignore"

        # 1.6 DECOY CLASSIFICATION: Track and classify potential decoys (Phase 6)
        if self.enable_decoys:
            for target in active_targets:
                if target.id not in self.state.decoy_tracking:
                    self.state.decoy_tracking[target.id] = self.state.sim_time
                time_tracked = self.state.sim_time - self.state.decoy_tracking[target.id]
                is_decoy, confidence = classify_threat(target, time_tracked)
                if is_decoy and confidence > 0.7:
                    target.engagement_decision = "ignore"
                    target.threat_type = "decoy"
                    self.state.engagement_stats["decoys_classified"] += 1

        # 2. GUIDANCE: Calculate acceleration for each interceptor
        # Track which interceptors are still active
        intercepted_interceptor_ids = set(pair[0] for pair in self.state.intercepted_pairs)

        # Track approved engagements for HMT human-in-loop mode
        # None = all engagements allowed (default when HMT disabled)
        approved_engagements = None
        if self.hmt:
            from .hmt import AuthorityLevel, ActionType, ActionStatus
            # In human-in-loop mode, only allow approved engagements
            if self.hmt.config.authority_level == AuthorityLevel.HUMAN_IN_LOOP:
                approved_engagements = set()  # Start with empty set
                for action in self.hmt.action_history:
                    if (action.action_type == ActionType.ENGAGE and
                        action.status in [ActionStatus.APPROVED, ActionStatus.AUTO_APPROVED]):
                        approved_engagements.add((action.entity_id, action.target_id))
            # In full_auto or human_on_loop, all engagements are allowed (None)

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

            # Use WTA assignment to get assigned target
            assigned_target = None
            assigned_target_id = None
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
                assigned_target_id = assigned_target.id

            # HMT human-in-loop check
            # If in human-in-loop mode and engagement not approved, coast instead
            if approved_engagements is not None and (interceptor.id, assigned_target_id) not in approved_engagements:
                # Not approved yet - coast (maintain velocity, no acceleration)
                interceptor.set_acceleration(Vec3.zero())
                continue

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

        # 3. ENVIRONMENT: Apply environmental forces
        if self.environment:
            self.environment.update(self.state.sim_time)
            for entity in self.state.targets + self.state.interceptors:
                env_accel = self.environment.get_total_environmental_acceleration(
                    velocity=entity.velocity,
                    position=entity.position,
                    cross_section=entity.cross_section,
                    mass=entity.mass,
                    drag_coefficient=entity.drag_coefficient,
                )
                entity.acceleration = entity.acceleration + env_accel

        # 4. COOPERATIVE: Check for zone-based handoffs
        if self.cooperative:
            self.cooperative.update(self.state.sim_time)

            # Check if any interceptor should hand off their target
            for interceptor in self.state.interceptors:
                if interceptor.id in intercepted_interceptor_ids:
                    continue

                # Get current assignment
                assigned_target_id = None
                if self.assignments:
                    assigned_target_id = self.assignments.get_target_for_interceptor(interceptor.id)

                if assigned_target_id:
                    # Find target position
                    target = next((t for t in self.state.targets if t.id == assigned_target_id), None)
                    if target:
                        # Check if handoff should be requested
                        handoff_suggestion = self.cooperative.should_request_handoff(
                            interceptor.id,
                            assigned_target_id,
                            target.position,
                            self.state.sim_time
                        )
                        if handoff_suggestion:
                            to_interceptor, reason = handoff_suggestion
                            self.cooperative.request_handoff(
                                from_interceptor=interceptor.id,
                                to_interceptor=to_interceptor,
                                target_id=assigned_target_id,
                                reason=reason,
                                current_time=self.state.sim_time
                            )

        # 5. SWARM: Apply swarm steering to interceptors
        if self.swarm:
            active_interceptors = [
                i for i in self.state.interceptors
                if i.id not in intercepted_interceptor_ids
            ]
            self.swarm.update(active_interceptors, dt)

            for interceptor in active_interceptors:
                swarm_accel = self.swarm.compute_steering(interceptor, active_interceptors)
                interceptor.acceleration = interceptor.acceleration + swarm_accel

        # 6. DATALINK: Update communication model
        if self.datalink:
            # Update entity positions in datalink
            for entity in self.state.targets + self.state.interceptors:
                self.datalink.update_entity_position(entity.id, entity.position)

            # Process message delivery
            delivered = self.datalink.update(self.state.sim_time)
            # Messages could be processed here for networked fire control

        # 7. HMT: Update human-machine teaming
        if self.hmt:
            timed_out = self.hmt.update(self.state.sim_time)
            # Handle timed out actions if needed

            # Propose engagement actions for human-in-loop mode
            from .hmt import AuthorityLevel, ActionType, PendingAction, ActionStatus
            if self.hmt.config.authority_level == AuthorityLevel.HUMAN_IN_LOOP:
                # Propose engagement for each active interceptor-target pair
                for interceptor in self.state.interceptors:
                    if interceptor.id in intercepted_interceptor_ids:
                        continue

                    # Get assigned target
                    assigned_target_id = None
                    if self.assignments:
                        assigned_target_id = self.assignments.get_target_for_interceptor(interceptor.id)

                    # Fallback: If no WTA assignment, use nearest target (same logic as guidance)
                    if assigned_target_id is None and active_targets:
                        nearest_target = min(
                            active_targets,
                            key=lambda t: interceptor.position.distance_to(t.position)
                        )
                        assigned_target_id = nearest_target.id

                    if assigned_target_id:
                        # Check if there's already a pending OR approved action for this pair
                        action_key = f"{interceptor.id}_{assigned_target_id}"
                        existing_pending = any(
                            a.entity_id == interceptor.id and a.target_id == assigned_target_id
                            for a in self.hmt.pending_actions.values()
                        )
                        # Also check if already approved in history
                        already_approved = any(
                            a.entity_id == interceptor.id and
                            a.target_id == assigned_target_id and
                            a.status in [ActionStatus.APPROVED, ActionStatus.AUTO_APPROVED]
                            for a in self.hmt.action_history
                        )

                        if not existing_pending and not already_approved:
                            # Calculate confidence based on geometry
                            target = next((t for t in active_targets if t.id == assigned_target_id), None)
                            if target:
                                dist = interceptor.position.distance_to(target.position)
                                # Higher confidence when closer
                                confidence = max(0.3, min(0.95, 1.0 - (dist / 5000)))

                                action = PendingAction.create(
                                    action_type=ActionType.ENGAGE,
                                    entity_id=interceptor.id,
                                    target_id=assigned_target_id,
                                    confidence=confidence,
                                    details={
                                        "range": dist,
                                        "action_key": action_key,
                                    },
                                    timestamp=self.state.sim_time,
                                    timeout=self.hmt.config.approval_timeout
                                )
                                self.hmt.propose_action(action)
            # Handle timed out actions if needed

        # 8. BATTERIES: Autonomous battery engagement loop
        if self.batteries:
            for battery in self.batteries:
                new_interceptors = battery.update(
                    targets=active_targets,
                    sim_time=self.state.sim_time,
                    existing_interceptors=self.state.interceptors,
                    impact_predictions=self.state.impact_predictions,
                )
                # Add newly launched interceptors to the simulation
                for interceptor in new_interceptors:
                    self.state.interceptors.append(interceptor)
                    # Create evasion state for new interceptor (none needed, but for consistency)
                    # Create WTA assignment for this battery-launched interceptor
                    target_id = None
                    track = battery.tracked_threats.get(None)
                    # Find which threat this interceptor was assigned to
                    for tid, trk in battery.tracked_threats.items():
                        if trk.assigned_interceptor == interceptor.id:
                            target_id = tid
                            break

                    if target_id:
                        new_assignment = Assignment(
                            interceptor_id=interceptor.id,
                            target_id=target_id,
                            cost=0.0,
                            reason="battery_auto"
                        )
                        if self.assignments:
                            self.assignments.assignments.append(new_assignment)
                        else:
                            self.assignments = AssignmentResult(
                                assignments=[new_assignment],
                                total_cost=0.0,
                                algorithm="battery_auto",
                                unassigned_interceptors=[],
                                unassigned_targets=[t.id for t in active_targets if t.id != target_id],
                                timestamp=self.state.sim_time
                            )

        # 9. LAUNCHERS: Update launch platforms and spawn new interceptors
        if self.state.launchers:
            for launcher in self.state.launchers:
                new_interceptors = launcher.update(
                    active_targets,
                    self.state.sim_time,
                    self.state.interceptors
                )
                # Add newly launched interceptors to the simulation
                for interceptor in new_interceptors:
                    self.state.interceptors.append(interceptor)
                    # Update WTA assignments to include new interceptor
                    # Find the target this interceptor was launched at
                    target_id = None
                    for event in launcher.launch_history:
                        if event.interceptor_id == interceptor.id:
                            target_id = event.target_id
                            break

                    if target_id:
                        # Create assignment for this interceptor
                        new_assignment = Assignment(
                            interceptor_id=interceptor.id,
                            target_id=target_id,
                            cost=0.0,  # Launched interceptors have committed
                            reason="launcher_auto"
                        )
                        # Add to existing assignments or create new AssignmentResult
                        if self.assignments:
                            self.assignments.assignments.append(new_assignment)
                        else:
                            self.assignments = AssignmentResult(
                                assignments=[new_assignment],
                                total_cost=0.0,
                                algorithm="launcher_auto",
                                unassigned_interceptors=[],
                                unassigned_targets=[t.id for t in active_targets if t.id != target_id],
                                timestamp=self.state.sim_time
                            )

        # 10. PHYSICS: Update all entities (skip intercepted ones)
        for target in self.state.targets:
            if target.id in intercepted_target_ids:
                # Stop intercepted targets
                target.velocity = Vec3.zero()
                target.set_acceleration(Vec3.zero())
                continue
            target.update(dt)
        for interceptor in self.state.interceptors:
            if interceptor.id in intercepted_interceptor_ids:
                # Already stopped above, but skip update too
                continue
            interceptor.update(dt)

        # 11. Update sim time
        self.state.sim_time += dt
        self.state.tick += 1

        # 12. Check end conditions
        self._check_end_conditions()

        # 13. Emit state event (include WTA assignments)
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
                final_miss = self.state.miss_distance if self.state.miss_distance != float('inf') else -1
                await asyncio.shield(self._emit_event({
                    "type": "complete",
                    "run_id": self.state.run_id,
                    "ts": time.time(),
                    "result": "stopped",
                    "final_miss_distance": final_miss,
                    "sim_time": self.state.sim_time,
                    "ticks": self.state.tick,
                }))
            except asyncio.CancelledError:
                pass  # Ignore if shield still gets cancelled

            raise  # Re-raise so the caller knows it was cancelled

        # Emit final event
        final_miss = self.state.miss_distance if self.state.miss_distance != float('inf') else -1
        await self._emit_event({
            "type": "complete",
            "run_id": self.state.run_id,
            "ts": time.time(),
            "result": self.state.result.value,
            "final_miss_distance": final_miss,
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
    # Multi-target scenarios
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
    # Swarm scenarios
    "swarm_v_formation": {
        "description": "4 interceptors in V-formation vs 2 targets",
        "target_start": Vec3(4000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.NONE,
        "num_interceptors": 4,
        "num_targets": 2,
        "target_spacing": 500.0,
        "interceptor_spacing": 150.0,
        "enable_swarm": True,
        "swarm_formation": "v_formation",
        "swarm_spacing": 200.0,
    },
    "swarm_echelon": {
        "description": "4 interceptors in echelon vs 2 weaving targets",
        "target_start": Vec3(4000, 0, 800),
        "target_velocity": Vec3(-100, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.WEAVE,
        "num_interceptors": 4,
        "num_targets": 2,
        "target_spacing": 400.0,
        "interceptor_spacing": 150.0,
        "enable_swarm": True,
        "swarm_formation": "echelon_right",
        "swarm_spacing": 180.0,
    },
    "swarm_line_abreast": {
        "description": "6 interceptors line abreast vs 3 targets",
        "target_start": Vec3(4500, 0, 900),
        "target_velocity": Vec3(-110, 0, -5),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(160, 0, 25),
        "evasion": EvasionType.NONE,
        "num_interceptors": 6,
        "num_targets": 3,
        "target_spacing": 350.0,
        "interceptor_spacing": 120.0,
        "enable_swarm": True,
        "swarm_formation": "line_abreast",
        "swarm_spacing": 250.0,
    },
    "swarm_diamond": {
        "description": "4 interceptors in diamond vs 1 evading target",
        "target_start": Vec3(3500, 0, 800),
        "target_velocity": Vec3(-90, 0, 0),
        "interceptor_start": Vec3(0, 0, 600),
        "interceptor_velocity": Vec3(150, 0, 20),
        "evasion": EvasionType.RANDOM_JINK,
        "num_interceptors": 4,
        "num_targets": 1,
        "enable_swarm": True,
        "swarm_formation": "diamond",
        "swarm_spacing": 200.0,
    },
    # ─── IRON DOME SCENARIOS ────────────────────────────────────────
    "iron_shield": {
        "description": "Qassam rockets vs Iron Dome — 3 rockets, only 2 threaten the city",
        "target_start": Vec3(8000, 0, 100),        # 8km east, near ground
        "target_velocity": Vec3(-450, 0, 300),      # West at 450m/s, up at 300m/s (ballistic arc)
        "interceptor_start": Vec3(-1000, 1000, 50), # Near city, offset for intercept geometry
        "interceptor_velocity": Vec3(200, 0, 300),  # Upward launch toward threat
        "evasion": EvasionType.NONE,
        "num_targets": 3,
        "target_spacing": 3500.0,
        "threat_type": "qassam",
        "interceptor_type": "tamir",
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Tel Aviv",
                center=Vec3(-2000, 1000, 0),  # Offset in Y so 2 rockets hit, 1 misses
                radius=3000.0,
                priority=1,
                population=400000,
            ),
        ],
    },
    "iron_dome_salvo": {
        "description": "5-rocket salvo against defended city",
        "target_start": Vec3(10000, 0, 100),
        "target_velocity": Vec3(-180, 20, 280),
        "interceptor_start": Vec3(0, 0, 50),
        "interceptor_velocity": Vec3(120, 0, 250),
        "evasion": EvasionType.NONE,
        "num_targets": 5,
        "target_spacing": 1200.0,
        "threat_type": "grad",
        "interceptor_type": "tamir",
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Ashkelon",
                center=Vec3(-1000, 0, 0),
                radius=2500.0,
                priority=1,
                population=150000,
            ),
            ProtectedArea(
                id="base_bravo",
                name="IDF Base",
                center=Vec3(-1000, 3000, 0),
                radius=1500.0,
                priority=2,
                population=5000,
            ),
        ],
    },
    # ─── BATTERY SCENARIOS (Phase 5) ────────────────────────────────
    "iron_dome_battery": {
        "description": "Single Iron Dome battery vs 5-rocket salvo — autonomous engagement",
        "target_start": Vec3(8000, 0, 100),
        "target_velocity": Vec3(-400, 0, 280),
        "interceptor_start": Vec3(-500, 500, 50),  # Interceptor start near battery (fallback)
        "interceptor_velocity": Vec3(200, 0, 300),
        "evasion": EvasionType.NONE,
        "num_targets": 5,
        "num_interceptors": 0,  # No pre-placed interceptors — battery launches them
        "target_spacing": 2000.0,
        "threat_type": "qassam",
        "interceptor_type": "tamir",
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Sderot",
                center=Vec3(-2000, 500, 0),
                radius=2500.0,
                priority=1,
                population=30000,
            ),
        ],
        "batteries": [
            BatteryConfig(
                position=Vec3(-500, 500, 0),
                name="Iron Dome Battery Alpha",
                tier="iron_dome",
                interceptor_type="tamir",
                radar_range=70000.0,
                radar_sector=360.0,
                num_launchers=3,
                missiles_per_launcher=20,
                max_simultaneous=6,
                min_range=4000.0,
                max_range=70000.0,
                launch_speed=250.0,
                launch_elevation=80.0,
                protected_areas=[
                    ProtectedArea(
                        id="city_alpha",
                        name="Sderot",
                        center=Vec3(-2000, 500, 0),
                        radius=2500.0,
                        priority=1,
                        population=30000,
                    ),
                ],
            ),
        ],
    },
    "iron_dome_dual_battery": {
        "description": "Two Iron Dome batteries defending two cities vs 8-rocket salvo",
        "target_start": Vec3(10000, 0, 100),
        "target_velocity": Vec3(-380, 30, 300),
        "interceptor_start": Vec3(-500, 0, 50),
        "interceptor_velocity": Vec3(200, 0, 300),
        "evasion": EvasionType.NONE,
        "num_targets": 8,
        "num_interceptors": 0,
        "target_spacing": 1800.0,
        "threat_type": "grad",
        "interceptor_type": "tamir",
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Ashkelon",
                center=Vec3(-1000, -1000, 0),
                radius=2500.0,
                priority=1,
                population=150000,
            ),
            ProtectedArea(
                id="city_bravo",
                name="Beer Sheva",
                center=Vec3(-1000, 4000, 0),
                radius=3000.0,
                priority=2,
                population=210000,
            ),
        ],
        "batteries": [
            BatteryConfig(
                position=Vec3(0, -500, 0),
                name="Battery Alpha",
                tier="iron_dome",
                interceptor_type="tamir",
                radar_range=70000.0,
                radar_sector=360.0,
                num_launchers=3,
                missiles_per_launcher=20,
                max_simultaneous=6,
                min_range=4000.0,
                max_range=70000.0,
                launch_speed=250.0,
                launch_elevation=80.0,
                protected_areas=[
                    ProtectedArea(
                        id="city_alpha",
                        name="Ashkelon",
                        center=Vec3(-1000, -1000, 0),
                        radius=2500.0,
                        priority=1,
                        population=150000,
                    ),
                ],
            ),
            BatteryConfig(
                position=Vec3(0, 3500, 0),
                name="Battery Bravo",
                tier="iron_dome",
                interceptor_type="tamir",
                radar_range=70000.0,
                radar_sector=360.0,
                num_launchers=3,
                missiles_per_launcher=20,
                max_simultaneous=6,
                min_range=4000.0,
                max_range=70000.0,
                launch_speed=250.0,
                launch_elevation=80.0,
                protected_areas=[
                    ProtectedArea(
                        id="city_bravo",
                        name="Beer Sheva",
                        center=Vec3(-1000, 4000, 0),
                        radius=3000.0,
                        priority=2,
                        population=210000,
                    ),
                ],
            ),
        ],
    },
    # ─── WAVE / SATURATION SCENARIOS (Phase 6) ────────────────────────
    "iron_rain": {
        "description": "3 waves of mixed rockets with decoys — saturation attack",
        "target_start": Vec3(8000, 0, 100),
        "target_velocity": Vec3(-400, 0, 280),
        "interceptor_start": Vec3(-500, 500, 50),
        "interceptor_velocity": Vec3(200, 0, 300),
        "evasion": EvasionType.NONE,
        "num_targets": 3,       # Initial wave (pre-placed)
        "num_interceptors": 0,  # Battery handles launches
        "target_spacing": 2000.0,
        "threat_type": "qassam",
        "interceptor_type": "tamir",
        "enable_decoys": True,
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Sderot",
                center=Vec3(-2000, 500, 0),
                radius=2500.0,
                priority=1,
                population=30000,
            ),
        ],
        "batteries": [
            BatteryConfig(
                position=Vec3(-500, 500, 0),
                name="Battery Alpha",
                tier="iron_dome",
                interceptor_type="tamir",
                radar_range=70000.0,
                num_launchers=3,
                missiles_per_launcher=20,
                max_simultaneous=6,
                min_range=4000.0,
                max_range=70000.0,
                launch_speed=250.0,
                launch_elevation=80.0,
                protected_areas=[
                    ProtectedArea(
                        id="city_alpha",
                        name="Sderot",
                        center=Vec3(-2000, 500, 0),
                        radius=2500.0,
                        priority=1,
                        population=30000,
                    ),
                ],
            ),
        ],
        "waves": [
            ThreatWave(
                wave_id=1,
                spawn_time=8.0,
                threat_type="grad",
                count=4,
                spacing=1500.0,
                start_position=Vec3(9000, -2000, 100),
                velocity=Vec3(-380, 40, 300),
                salvo_interval=0.8,
            ),
            ThreatWave(
                wave_id=2,
                spawn_time=18.0,
                threat_type="qassam",
                count=5,
                spacing=1800.0,
                start_position=Vec3(8500, 1000, 100),
                velocity=Vec3(-420, -20, 260),
                salvo_interval=0.5,
            ),
        ],
    },
    "last_stand": {
        "description": "Continuous waves overwhelming a single battery — fight to ammo depletion",
        "target_start": Vec3(9000, 0, 100),
        "target_velocity": Vec3(-420, 0, 300),
        "interceptor_start": Vec3(-500, 0, 50),
        "interceptor_velocity": Vec3(200, 0, 300),
        "evasion": EvasionType.NONE,
        "num_targets": 4,
        "num_interceptors": 0,
        "target_spacing": 1500.0,
        "threat_type": "qassam",
        "interceptor_type": "tamir",
        "protected_areas": [
            ProtectedArea(
                id="city_alpha",
                name="Ashkelon",
                center=Vec3(-1500, 0, 0),
                radius=2500.0,
                priority=1,
                population=150000,
            ),
        ],
        "batteries": [
            BatteryConfig(
                position=Vec3(0, 0, 0),
                name="Battery Alpha",
                tier="iron_dome",
                interceptor_type="tamir",
                radar_range=70000.0,
                num_launchers=3,
                missiles_per_launcher=10,  # Reduced ammo — pressure test
                max_simultaneous=6,
                min_range=4000.0,
                max_range=70000.0,
                launch_speed=250.0,
                launch_elevation=80.0,
                protected_areas=[
                    ProtectedArea(
                        id="city_alpha",
                        name="Ashkelon",
                        center=Vec3(-1500, 0, 0),
                        radius=2500.0,
                        priority=1,
                        population=150000,
                    ),
                ],
            ),
        ],
        "waves": [
            ThreatWave(
                wave_id=1,
                spawn_time=6.0,
                threat_type="grad",
                count=5,
                spacing=1200.0,
                start_position=Vec3(9500, -1500, 100),
                velocity=Vec3(-400, 30, 280),
                salvo_interval=0.6,
            ),
            ThreatWave(
                wave_id=2,
                spawn_time=14.0,
                threat_type="qassam",
                count=6,
                spacing=1400.0,
                start_position=Vec3(8500, 1500, 100),
                velocity=Vec3(-430, -20, 290),
                salvo_interval=0.4,
            ),
            ThreatWave(
                wave_id=3,
                spawn_time=22.0,
                threat_type="grad",
                count=8,
                spacing=1000.0,
                start_position=Vec3(10000, 0, 100),
                velocity=Vec3(-450, 0, 310),
                salvo_interval=0.3,
            ),
            ThreatWave(
                wave_id=4,
                spawn_time=32.0,
                threat_type="qassam",
                count=10,
                spacing=800.0,
                start_position=Vec3(9000, 500, 100),
                velocity=Vec3(-440, -10, 300),
                salvo_interval=0.2,
            ),
        ],
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
    - waves: List of ThreatWave configs (optional, Phase 6)
    - batteries: List of BatteryConfig (optional, Phase 5)
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
