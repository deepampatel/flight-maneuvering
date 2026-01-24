"""
API Server - The Bridge Between Sim and UI

This server does three things:
1. REST API: Start/stop runs, get run info
2. WebSocket: Stream real-time sim events to UI
3. Orchestration: Manage the simulation lifecycle

Architecture insight:
In production systems, you'd have separate services.
Here we keep it simple but maintain the INTERFACES
that would allow separation later.
"""

from __future__ import annotations
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Set, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from sim.engine import SimEngine, SimConfig, load_scenario, SCENARIOS
from sim.vector import Vec3
from sim.guidance import GuidanceType, GuidanceParams, create_guidance_function
from sim.monte_carlo import MonteCarloConfig, run_monte_carlo, parameter_sweep
from sim.evasion import EvasionType, EvasionConfig
from sim.envelope import EnvelopeConfig, compute_engagement_envelope
from sim.intercept import compute_intercept_geometry, compute_all_geometries, compute_all_geometries_multi
from sim.threat import (
    ThreatWeights, compute_threat_score, assess_all_threats, quick_threat_assessment,
    ml_threat_assessment, hybrid_threat_assessment
)
from sim.recording import (
    RecordingManager, ReplayEngine, ReplayConfig,
    EngagementRecording, get_recording_manager
)
from sim.sensor import SensorModel, SensorConfig, create_sensor_state
from sim.assignment import (
    WTAAlgorithm, compute_assignment, compute_cost_matrix,
    greedy_nearest_assignment, greedy_threat_assignment, hungarian_assignment
)
from sim.environment import EnvironmentConfig, EnvironmentModel, create_wind_from_speed_direction
from sim.kalman import KalmanFilter, KalmanState, KalmanConfig
from sim.fusion import TrackFusionManager, FusionConfig, FusedTrack
from sim.cooperation import (
    CooperativeEngagementManager, EngagementZone, HandoffRequest,
    HandoffStatus, HandoffReason
)
from sim.ml import (
    ThreatModel, GuidanceModel, MLConfig,
    get_model_registry, extract_threat_features,
)
from sim.ml.inference import ONNX_AVAILABLE


# ============================================================
# Connection Manager - Handles multiple WebSocket clients
# ============================================================

class ConnectionManager:
    """
    Manages WebSocket connections.

    Why a manager?
    - Multiple UIs can connect (imagine ops console + big screen)
    - Clean disconnect handling
    - Broadcast to all clients
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send to all connected clients."""
        if not self.active_connections:
            return

        data = json.dumps(message)
        # Send to all, handle disconnects
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


# Global instances
manager = ConnectionManager()
current_engine: Optional[SimEngine] = None
run_task: Optional[asyncio.Task] = None
recording_manager = get_recording_manager()
current_replay_engine: Optional[ReplayEngine] = None
replay_task: Optional[asyncio.Task] = None

# Phase 6: Global sensor states for track management
sensor_states: dict = {}  # interceptor_id -> SensorState


# ============================================================
# API Models
# ============================================================

class Vec3Model(BaseModel):
    """3D vector for API."""
    x: float
    y: float
    z: float


class PlannedEntityModel(BaseModel):
    """Planned entity from mission planner."""
    id: str
    type: str  # "interceptor" or "target"
    position: Vec3Model
    velocity: Vec3Model


class PlannedZoneModel(BaseModel):
    """Planned zone from mission planner."""
    id: str
    name: str
    center: Vec3Model
    dimensions: Vec3Model
    color: str = "#00ff00"


class RunConfig(BaseModel):
    """Request body for starting a run."""
    scenario: str = "head_on"
    real_time: bool = True
    dt: float = 0.02
    max_time: float = 60.0
    kill_radius: float = 150.0  # Proximity fuse radius
    guidance: str = "proportional_nav"  # pure_pursuit, proportional_nav, augmented_pn
    nav_constant: float = 4.0
    evasion: str = "none"  # none, constant_turn, weave, barrel_roll, random_jink
    num_interceptors: int = 1  # Number of interceptors (unlimited)
    # Phase 5: Multi-target support
    num_targets: Optional[int] = None  # Override scenario's num_targets if set
    # Phase 5: WTA algorithm
    wta_algorithm: str = "hungarian"  # greedy_nearest, greedy_threat, hungarian, round_robin
    # Phase 6: Environment configuration
    wind_speed: float = 0.0  # Wind speed in m/s
    wind_direction: float = 0.0  # Wind direction in degrees (0=North, 90=East)
    wind_gusts: float = 0.0  # Gust amplitude in m/s
    enable_drag: bool = False  # Enable aerodynamic drag
    # Phase 6: Cooperative engagement
    enable_cooperative: bool = False  # Enable cooperative engagement features
    # Mission Planner: Custom entities
    custom_entities: Optional[list[PlannedEntityModel]] = None
    custom_zones: Optional[list[PlannedZoneModel]] = None


class MonteCarloRequest(BaseModel):
    """Request body for Monte Carlo batch run."""
    scenario: str = "head_on"
    guidance: str = "proportional_nav"
    nav_constant: float = 4.0
    num_runs: int = 100
    kill_radius: float = 50.0
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0


class ParameterSweepRequest(BaseModel):
    """Request body for parameter sweep."""
    scenario: str = "head_on"
    guidance: str = "proportional_nav"
    param_name: str = "nav_constant"  # nav_constant, kill_radius, position_noise_std
    param_values: list = [2.0, 3.0, 4.0, 5.0, 6.0]
    num_runs_per_value: int = 50
    kill_radius: float = 50.0


class EnvelopeRequest(BaseModel):
    """Request body for engagement envelope analysis."""
    # Range sweep (meters)
    range_min: float = 1000.0
    range_max: float = 5000.0
    range_steps: int = 8

    # Bearing sweep (degrees)
    bearing_min: float = -90.0
    bearing_max: float = 90.0
    bearing_steps: int = 10

    # Elevation sweep (degrees)
    elevation_min: float = -20.0
    elevation_max: float = 20.0
    elevation_steps: int = 5

    # Monte Carlo settings
    runs_per_point: int = 5

    # Guidance settings
    guidance: str = "proportional_nav"
    nav_constant: float = 4.0
    kill_radius: float = 50.0

    # Target settings
    target_speed: float = 100.0
    evasion: str = "none"

    # Interceptor settings
    interceptor_speed: float = 200.0


class RunStatus(BaseModel):
    """Response with current run status."""
    run_id: Optional[str]
    status: str
    sim_time: float
    result: Optional[str]


class ThreatWeightsRequest(BaseModel):
    """Custom weights for threat scoring."""
    time_to_impact: float = 0.35
    closing_velocity: float = 0.25
    aspect_angle: float = 0.20
    altitude_advantage: float = 0.10
    maneuverability: float = 0.10


class ReplayConfigRequest(BaseModel):
    """Configuration for replay playback."""
    speed_multiplier: float = 1.0
    start_tick: int = 0
    end_tick: Optional[int] = None


class RecordingStartRequest(BaseModel):
    """Request to start recording."""
    scenario_name: Optional[str] = None


# Phase 5: Sensor Configuration
class SensorConfigRequest(BaseModel):
    """Request body for sensor configuration."""
    max_range: float = 10000.0
    min_range: float = 100.0
    field_of_view: float = 120.0
    detection_probability: float = 0.95
    range_noise_std: float = 50.0
    angle_noise_std: float = 1.0


# Phase 5: WTA Configuration
class WTAConfigRequest(BaseModel):
    """Request body for WTA algorithm configuration."""
    algorithm: str = "greedy_nearest"  # greedy_nearest, greedy_threat, hungarian, round_robin


# Phase 6: Cooperative Engagement Models
class EngagementZoneRequest(BaseModel):
    """Request body for creating an engagement zone."""
    name: str = "Zone Alpha"
    center_x: float = 1500.0
    center_y: float = 0.0
    center_z: float = 600.0
    width: float = 1000.0  # x dimension
    depth: float = 1000.0  # y dimension
    height: float = 500.0  # z dimension
    rotation: float = 0.0  # heading in degrees
    priority: int = 1
    color: str = "#00ff00"


class HandoffRequestModel(BaseModel):
    """Request body for requesting a target handoff."""
    from_interceptor: str
    to_interceptor: str
    target_id: str
    reason: str = "manual"  # manual, fuel_low, out_of_envelope, zone_boundary, better_geometry


class InterceptorZoneAssignment(BaseModel):
    """Request body for assigning an interceptor to a zone."""
    interceptor_id: str
    zone_id: str


# ============================================================
# FastAPI App
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown logic."""
    print("🚀 Air Dominance Sim Server starting...")
    yield
    print("👋 Server shutting down...")
    # Cancel any running simulation
    global run_task
    if run_task and not run_task.done():
        run_task.cancel()


app = FastAPI(
    title="Air Dominance Simulation",
    description="Real-time air intercept simulation sandbox",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local dev (React on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REST Endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Air Dominance Simulation",
        "version": "0.1.0",
        "status": "ready",
    }


@app.get("/scenarios")
async def list_scenarios():
    """List available scenarios."""
    return {
        name: {
            "name": name,
            "description": info["description"],
            "evasion": info.get("evasion", "none").value if hasattr(info.get("evasion", "none"), "value") else "none",
            # Phase 5: Include multi-target info
            "num_targets": info.get("num_targets", 1),
            "target_spacing": info.get("target_spacing", 300.0),
        }
        for name, info in SCENARIOS.items()
    }


@app.get("/evasion")
async def list_evasion_types():
    """List available evasion maneuvers."""
    return {
        "evasion_types": [
            {"id": "none", "name": "None", "description": "Target flies straight, no evasion"},
            {"id": "constant_turn", "name": "Constant Turn", "description": "Sustained turn at fixed rate"},
            {"id": "weave", "name": "Weave (S-Turns)", "description": "Periodic direction reversals"},
            {"id": "barrel_roll", "name": "Barrel Roll", "description": "3D spiral evasion maneuver"},
            {"id": "random_jink", "name": "Random Jink", "description": "Unpredictable random direction changes"},
        ]
    }


@app.post("/runs")
async def start_run(config: RunConfig):
    """
    Start a new simulation run.

    This:
    1. Creates a new SimEngine
    2. Sets up the scenario
    3. Starts the sim in a background task
    4. Returns immediately with run_id
    """
    global current_engine, run_task

    # Cancel any existing run
    if run_task and not run_task.done():
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    # Load scenario
    try:
        scenario = load_scenario(config.scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create guidance function
    guidance_type = GuidanceType(config.guidance)
    guidance_params = GuidanceParams(nav_constant=config.nav_constant)
    guidance = create_guidance_function(guidance_type, guidance_params)

    # Determine evasion type (from config or scenario default)
    if config.evasion != "none":
        evasion_type = EvasionType(config.evasion)
    elif "evasion" in scenario:
        evasion_type = scenario["evasion"]
    else:
        evasion_type = EvasionType.NONE

    # Determine WTA algorithm
    try:
        wta_algorithm = WTAAlgorithm(config.wta_algorithm)
    except ValueError:
        wta_algorithm = WTAAlgorithm.HUNGARIAN

    # Phase 6: Create environment config if wind or drag enabled
    environment_config = None
    if config.wind_speed > 0 or config.enable_drag:
        wind_velocity = create_wind_from_speed_direction(config.wind_speed, config.wind_direction)
        environment_config = EnvironmentConfig(
            wind_velocity=wind_velocity,
            wind_gust_amplitude=config.wind_gusts,
            enable_drag=config.enable_drag,
        )

    # Create engine with config
    sim_config = SimConfig(
        dt=config.dt,
        max_time=config.max_time,
        kill_radius=config.kill_radius,
        real_time=config.real_time,
        enable_cooperative=config.enable_cooperative,
    )
    current_engine = SimEngine(
        config=sim_config,
        guidance=guidance,
        evasion_type=evasion_type,
        wta_algorithm=wta_algorithm,
        environment_config=environment_config,
        enable_cooperative=config.enable_cooperative,
    )

    # Register broadcast handler
    current_engine.on_event(manager.broadcast)

    # Register recording handler if recording is active
    async def recording_handler(event: dict):
        if recording_manager.is_recording and event.get("type") == "state":
            # Extract state from event for recording
            recording_manager.record_frame(
                tick=event.get("tick", 0),
                sim_time=event.get("sim_time", 0.0),
                target=current_engine.state.target,
                interceptors=current_engine.state.interceptors,
            )
    current_engine.on_event(recording_handler)

    # Check if using custom entities from mission planner
    if config.custom_entities and len(config.custom_entities) > 0:
        # Use custom entity setup
        current_engine.setup_custom_scenario(config.custom_entities)
    else:
        # Setup scenario with multi-target support (Phase 5)
        num_targets = config.num_targets if config.num_targets else scenario.get("num_targets", 1)
        target_spacing = scenario.get("target_spacing", 300.0)

        current_engine.setup_scenario(
            target_start=scenario["target_start"],
            target_velocity=scenario["target_velocity"],
            interceptor_start=scenario["interceptor_start"],
            interceptor_velocity=scenario["interceptor_velocity"],
            num_interceptors=config.num_interceptors,
            num_targets=num_targets,
            target_spacing=target_spacing,
        )

    # Create custom zones if provided
    if config.custom_zones and current_engine.cooperative:
        for zone in config.custom_zones:
            current_engine.cooperative.create_zone(
                name=zone.name,
                center=Vec3(zone.center.x, zone.center.y, zone.center.z),
                dimensions=Vec3(zone.dimensions.x, zone.dimensions.y, zone.dimensions.z),
                color=zone.color,
            )

    # Start simulation in background
    run_task = asyncio.create_task(current_engine.run())

    return {
        "run_id": current_engine.state.run_id,
        "scenario": config.scenario if not config.custom_entities else "custom",
        "status": "started",
        "num_interceptors": len([e for e in (config.custom_entities or []) if e.type == "interceptor"]) or config.num_interceptors,
        "num_targets": len([e for e in (config.custom_entities or []) if e.type == "target"]) or (config.num_targets or 1),
    }


@app.get("/runs/current")
async def get_current_run():
    """Get status of current run."""
    if current_engine is None or current_engine.state is None:
        return {"status": "no_run"}

    state = current_engine.state
    return {
        "run_id": state.run_id,
        "status": state.status.value,
        "sim_time": state.sim_time,
        "tick": state.tick,
        "result": state.result.value,
        "miss_distance": state.miss_distance,
    }


@app.post("/runs/stop")
async def stop_run():
    """Stop the current run."""
    global run_task

    if run_task and not run_task.done():
        run_task.cancel()
        # Wait for the task to actually finish so events are emitted
        try:
            await run_task
        except asyncio.CancelledError:
            pass  # Expected
        return {"status": "stopped"}

    return {"status": "no_run_active"}


# ============================================================
# Monte Carlo Endpoints
# ============================================================

@app.get("/guidance")
async def list_guidance():
    """List available guidance laws."""
    return {
        "guidance_laws": [
            {"id": "pure_pursuit", "name": "Pure Pursuit", "description": "Point at target's current position"},
            {"id": "proportional_nav", "name": "Proportional Navigation", "description": "Industry standard - drives LOS rate to zero"},
            {"id": "augmented_pn", "name": "Augmented PN", "description": "PN with target acceleration compensation"},
        ]
    }


@app.post("/monte-carlo")
async def run_monte_carlo_batch(request: MonteCarloRequest):
    """
    Run a Monte Carlo batch analysis.

    Returns statistics on intercept success rate, miss distance distribution, etc.
    """
    try:
        scenario = load_scenario(request.scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = MonteCarloConfig(
        target_start=scenario["target_start"],
        target_velocity=scenario["target_velocity"],
        interceptor_start=scenario["interceptor_start"],
        interceptor_velocity=scenario["interceptor_velocity"],
        guidance_type=GuidanceType(request.guidance),
        nav_constant=request.nav_constant,
        num_runs=min(request.num_runs, 1000),  # Cap at 1000
        kill_radius=request.kill_radius,
        position_noise_std=request.position_noise_std,
        velocity_noise_std=request.velocity_noise_std,
    )

    results = await run_monte_carlo(config)
    return results.to_dict()


@app.post("/monte-carlo/sweep")
async def run_parameter_sweep_endpoint(request: ParameterSweepRequest):
    """
    Run a parameter sweep analysis.

    Varies one parameter across multiple values to find optimal settings.
    """
    try:
        scenario = load_scenario(request.scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    base_config = MonteCarloConfig(
        target_start=scenario["target_start"],
        target_velocity=scenario["target_velocity"],
        interceptor_start=scenario["interceptor_start"],
        interceptor_velocity=scenario["interceptor_velocity"],
        guidance_type=GuidanceType(request.guidance),
        num_runs=min(request.num_runs_per_value, 200),  # Cap per value
        kill_radius=request.kill_radius,
    )

    results = await parameter_sweep(
        base_config,
        request.param_name,
        request.param_values,
    )

    return {
        "param_name": request.param_name,
        "results": [r.to_dict() for r in results],
    }


@app.post("/envelope")
async def run_envelope_analysis(request: EnvelopeRequest):
    """
    Compute engagement envelope.

    Sweeps across range, bearing, and elevation to find the intercept probability
    at each point. Returns both 2D heatmap and 3D surface data.
    """
    config = EnvelopeConfig(
        range_min=request.range_min,
        range_max=request.range_max,
        range_steps=min(request.range_steps, 15),  # Cap for performance
        bearing_min=request.bearing_min,
        bearing_max=request.bearing_max,
        bearing_steps=min(request.bearing_steps, 20),
        elevation_min=request.elevation_min,
        elevation_max=request.elevation_max,
        elevation_steps=min(request.elevation_steps, 10),
        runs_per_point=min(request.runs_per_point, 20),
        guidance_type=GuidanceType(request.guidance),
        nav_constant=request.nav_constant,
        kill_radius=request.kill_radius,
        target_speed=request.target_speed,
        evasion_type=EvasionType(request.evasion),
        interceptor_speed=request.interceptor_speed,
    )

    results = await compute_engagement_envelope(config)
    return results.to_dict()


# ============================================================
# Intercept Geometry Endpoints
# ============================================================

@app.get("/intercept-geometry")
async def get_intercept_geometry():
    """
    Get current intercept geometry from running simulation.

    Returns geometry for all interceptor-target pairs including:
    - LOS range and rate
    - Aspect and antenna train angles
    - Lead angle and collision course status
    - Time to intercept prediction

    Phase 5: Updated to support multiple targets.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state

    # Phase 5: Use multi-target geometry computation
    if len(state.targets) > 1:
        geometries = compute_all_geometries_multi(state.interceptors, state.targets)
    else:
        # Backward compatible for single target
        geometries = compute_all_geometries(state.interceptors, state.target)

    return {
        "timestamp": state.sim_time,
        "num_targets": len(state.targets),
        "num_interceptors": len(state.interceptors),
        "geometries": [g.to_dict() for g in geometries]
    }


# ============================================================
# Threat Assessment Endpoints
# ============================================================

@app.get("/threat-assessment")
async def get_threat_assessment(interceptor_id: str = None):
    """
    Get threat assessment from current simulation state.

    Returns threat scores for all targets ranked by priority.
    Optionally filter by interceptor_id.

    Phase 5: Updated to support multiple targets.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state
    assessments = []

    # Get interceptors to assess from
    interceptors = state.interceptors
    if interceptor_id:
        interceptors = [i for i in interceptors if i.id == interceptor_id]
        if not interceptors:
            raise HTTPException(status_code=404, detail=f"Interceptor {interceptor_id} not found")

    for interceptor in interceptors:
        # Compute geometry for all targets
        geometries = [
            compute_intercept_geometry(interceptor, target)
            for target in state.targets
        ]
        # Then threat assessment for all targets
        assessment = assess_all_threats(
            interceptor,
            state.targets,
            geometries
        )
        assessments.append(assessment.to_dict())

    return {
        "num_targets": len(state.targets),
        "assessments": assessments
    }


@app.post("/threat-assessment/configure")
async def configure_threat_weights(weights: ThreatWeightsRequest):
    """
    Configure custom threat weights (for testing/tuning).

    Weights should sum to 1.0 (will be normalized if not).
    """
    return {
        "status": "ok",
        "normalized_weights": {
            "time_to_impact": weights.time_to_impact,
            "closing_velocity": weights.closing_velocity,
            "aspect_angle": weights.aspect_angle,
            "altitude_advantage": weights.altitude_advantage,
            "maneuverability": weights.maneuverability
        }
    }


# ============================================================
# Recording Endpoints
# ============================================================

@app.post("/recordings/start")
async def start_recording(request: RecordingStartRequest = None):
    """
    Start recording the current/next simulation.

    Recording will capture all state changes until stopped.
    """
    global recording_manager

    if recording_manager.is_recording:
        return {"status": "already_recording", "recording_id": recording_manager.active_recording.recording_id}

    # Determine scenario name
    scenario_name = "unknown"
    config = {}

    if request and request.scenario_name:
        scenario_name = request.scenario_name
    elif current_engine and current_engine.state:
        # Try to get from current engine
        scenario_name = "live_capture"

    if current_engine:
        config = {
            "guidance_type": "unknown",  # Would need to track this
            "evasion_type": "unknown",
            "dt": current_engine.config.dt if current_engine.config else 0.02
        }

    recording_id = recording_manager.start_recording(scenario_name, config)

    return {
        "status": "recording_started",
        "recording_id": recording_id
    }


@app.post("/recordings/stop")
async def stop_recording():
    """
    Stop recording and save the engagement.
    """
    global recording_manager

    if not recording_manager.is_recording:
        return {"status": "not_recording"}

    # Get final state from engine if available
    result = "unknown"
    miss_distance = 0.0
    sim_time = 0.0

    if current_engine and current_engine.state:
        result = current_engine.state.result.value
        miss_distance = current_engine.state.miss_distance
        sim_time = current_engine.state.sim_time

    recording = recording_manager.stop_recording(result, miss_distance, sim_time)

    if recording:
        filepath = recording_manager.save_recording(recording)
        return {
            "status": "recording_saved",
            "recording_id": recording.recording_id,
            "filepath": filepath,
            "total_frames": len(recording.frames)
        }

    return {"status": "error", "message": "Failed to stop recording"}


@app.get("/recordings")
async def list_recordings():
    """List all saved recordings."""
    global recording_manager
    recordings = recording_manager.list_recordings()
    return {
        "recordings": recordings,
        "total": len(recordings)
    }


@app.get("/recordings/{recording_id}")
async def get_recording(recording_id: str):
    """Get recording details and metadata."""
    global recording_manager
    recording = recording_manager.load_recording(recording_id)

    if not recording:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

    return recording.to_metadata()


@app.delete("/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording."""
    global recording_manager

    if recording_manager.delete_recording(recording_id):
        return {"status": "deleted", "recording_id": recording_id}

    raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")


# ============================================================
# Replay Endpoints
# ============================================================

@app.post("/replay/{recording_id}/start")
async def start_replay(recording_id: str, config: ReplayConfigRequest = None):
    """
    Start replaying a recording.

    Events are streamed via WebSocket just like live simulation.
    """
    global current_replay_engine, replay_task, recording_manager

    # Stop any existing replay
    if replay_task and not replay_task.done():
        replay_task.cancel()
        try:
            await replay_task
        except asyncio.CancelledError:
            pass

    # Load recording
    recording = recording_manager.load_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

    # Create replay config
    replay_config = ReplayConfig(
        speed_multiplier=config.speed_multiplier if config else 1.0,
        start_tick=config.start_tick if config else 0,
        end_tick=config.end_tick if config else None
    )

    # Create replay engine
    current_replay_engine = ReplayEngine(recording, replay_config)
    current_replay_engine.on_event(manager.broadcast)

    # Start replay in background
    replay_task = asyncio.create_task(current_replay_engine.play())

    return {
        "status": "replay_started",
        "recording_id": recording_id,
        "total_ticks": len(recording.frames),
        "speed_multiplier": replay_config.speed_multiplier
    }


@app.post("/replay/pause")
async def pause_replay():
    """Pause current replay."""
    global current_replay_engine

    if current_replay_engine and current_replay_engine.is_playing:
        await current_replay_engine.pause()
        return {"status": "paused"}

    return {"status": "no_replay_active"}


@app.post("/replay/resume")
async def resume_replay():
    """Resume paused replay."""
    global current_replay_engine

    if current_replay_engine and current_replay_engine.is_paused:
        await current_replay_engine.resume()
        return {"status": "resumed"}

    return {"status": "no_paused_replay"}


@app.post("/replay/seek")
async def seek_replay(tick: int):
    """Seek to specific tick in replay."""
    global current_replay_engine

    if current_replay_engine:
        await current_replay_engine.seek(tick)
        return {"status": "seeked", "tick": tick}

    return {"status": "no_replay_active"}


@app.post("/replay/stop")
async def stop_replay():
    """Stop current replay."""
    global current_replay_engine, replay_task

    if replay_task and not replay_task.done():
        replay_task.cancel()
        try:
            await replay_task
        except asyncio.CancelledError:
            pass

    if current_replay_engine:
        await current_replay_engine.stop()
        current_replay_engine = None

    return {"status": "stopped"}


@app.get("/replay/state")
async def get_replay_state():
    """Get current replay state."""
    global current_replay_engine

    if current_replay_engine:
        return current_replay_engine.get_state()

    return {"status": "no_replay_active"}


# ============================================================
# Phase 5: Sensor Endpoints
# ============================================================

@app.get("/sensor/config")
async def get_sensor_config():
    """Get current sensor configuration."""
    config = SensorConfig()
    return {
        "max_range": config.max_range,
        "min_range": config.min_range,
        "field_of_view": config.field_of_view,
        "detection_probability": config.detection_probability,
        "range_noise_std": config.range_noise_std,
        "angle_noise_std": config.angle_noise_std,
        "update_rate": config.update_rate,
    }


@app.get("/sensor/detections")
async def get_sensor_detections():
    """
    Get simulated sensor detections for all interceptors.

    Returns what each interceptor's sensor would detect given current state.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state
    sensor = SensorModel()  # Use default config
    detections_by_interceptor = {}

    for interceptor in state.interceptors:
        detections = []
        for target in state.targets:
            detection = sensor.compute_detection(
                sensor_pos=interceptor.position,
                sensor_vel=interceptor.velocity,
                target=target,
                sim_time=state.sim_time
            )
            detections.append({
                "target_id": detection.target_id,
                "detected": detection.detected,
                "in_fov": detection.in_fov,
                "true_range": round(detection.true_range, 1),
                "measured_range": round(detection.measured_range, 1),
                "bearing": round(detection.bearing, 1),
                "elevation": round(detection.elevation, 1),
                "confidence": round(detection.confidence, 3),
                "estimated_position": detection.estimated_position.to_dict() if detection.detected else None,
            })
        detections_by_interceptor[interceptor.id] = detections

    return {
        "timestamp": state.sim_time,
        "detections": detections_by_interceptor
    }


# ============================================================
# Phase 5: Weapon-Target Assignment Endpoints
# ============================================================

@app.get("/wta/algorithms")
async def list_wta_algorithms():
    """List available WTA algorithms."""
    return {
        "algorithms": [
            {"id": "greedy_nearest", "name": "Greedy Nearest", "description": "Each interceptor takes nearest unassigned target"},
            {"id": "greedy_threat", "name": "Greedy Threat", "description": "Prioritize highest-threat targets first"},
            {"id": "hungarian", "name": "Hungarian (Optimal)", "description": "Optimal assignment minimizing total cost"},
            {"id": "round_robin", "name": "Round Robin", "description": "Simple sequential assignment"},
        ]
    }


@app.get("/wta/assignments")
async def get_current_assignments(algorithm: str = "greedy_nearest"):
    """
    Get weapon-target assignments for current simulation state.

    Args:
        algorithm: WTA algorithm to use (greedy_nearest, greedy_threat, hungarian, round_robin)
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state

    # Validate algorithm
    try:
        wta_algorithm = WTAAlgorithm(algorithm)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

    # Compute geometries and threats for better assignment
    geometries = compute_all_geometries_multi(state.interceptors, state.targets)

    threats = []
    for interceptor in state.interceptors:
        interceptor_geos = [g for g in geometries if g.interceptor_id == interceptor.id]
        assessment = assess_all_threats(interceptor, state.targets, interceptor_geos)
        threats.extend(assessment.threats)

    # Compute assignment
    result = compute_assignment(
        state.interceptors,
        state.targets,
        wta_algorithm,
        geometries,
        threats
    )

    return result.to_dict()


@app.post("/wta/configure")
async def configure_wta(config: WTAConfigRequest):
    """Configure WTA algorithm for simulation."""
    try:
        WTAAlgorithm(config.algorithm)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {config.algorithm}")

    return {
        "status": "ok",
        "algorithm": config.algorithm
    }


@app.get("/wta/cost-matrix")
async def get_cost_matrix():
    """
    Get the cost matrix for current interceptor-target assignments.

    Useful for visualization and debugging WTA decisions.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state
    geometries = compute_all_geometries_multi(state.interceptors, state.targets)

    cost_matrix = compute_cost_matrix(state.interceptors, state.targets, geometries)

    return {
        "timestamp": state.sim_time,
        "interceptor_ids": [i.id for i in state.interceptors],
        "target_ids": [t.id for t in state.targets],
        "cost_matrix": [[round(c, 3) for c in row] for row in cost_matrix]
    }


# ============================================================
# Phase 6: Environment Endpoints
# ============================================================

class EnvironmentConfigRequest(BaseModel):
    """Request body for environment configuration."""
    wind_speed: float = 0.0  # Wind speed in m/s
    wind_direction: float = 0.0  # Wind direction in degrees (0=North, 90=East)
    wind_gust_amplitude: float = 0.0  # Gust amplitude in m/s
    wind_gust_period: float = 5.0  # Gust period in seconds
    enable_drag: bool = False  # Enable aerodynamic drag
    drag_coefficient: float = 0.3  # Reference drag coefficient


@app.get("/environment/config")
async def get_environment_config():
    """Get current environment configuration."""
    if current_engine is None or current_engine.environment is None:
        return {
            "enabled": False,
            "wind_velocity": {"x": 0, "y": 0, "z": 0},
            "wind_gust_amplitude": 0.0,
            "wind_gust_period": 5.0,
            "enable_drag": False,
            "drag_coefficient": 0.3,
            "sea_level_density": 1.225,
        }

    env = current_engine.environment
    return {
        "enabled": True,
        "wind_velocity": env.config.wind_velocity.to_dict(),
        "wind_gust_amplitude": env.config.wind_gust_amplitude,
        "wind_gust_period": env.config.wind_gust_period,
        "enable_drag": env.config.enable_drag,
        "drag_coefficient": env.config.reference_drag_coefficient,
        "sea_level_density": env.config.sea_level_density,
        "current_wind": env.state.current_wind.to_dict() if env.state else {"x": 0, "y": 0, "z": 0},
    }


@app.get("/environment/state")
async def get_environment_state():
    """Get current environment state (time-varying values)."""
    if current_engine is None or current_engine.environment is None:
        raise HTTPException(status_code=400, detail="No active simulation with environment")

    env = current_engine.environment
    state = current_engine.state

    return {
        "timestamp": state.sim_time if state else 0.0,
        "current_wind": env.state.current_wind.to_dict(),
        "wind_speed": env.state.current_wind.magnitude(),
    }


@app.post("/environment/configure")
async def configure_environment(config: EnvironmentConfigRequest):
    """
    Configure environment for next simulation run.

    Note: This returns the configuration that will be used.
    The actual environment is created when a run starts.
    """
    wind_velocity = create_wind_from_speed_direction(config.wind_speed, config.wind_direction)

    return {
        "status": "ok",
        "config": {
            "wind_velocity": wind_velocity.to_dict(),
            "wind_gust_amplitude": config.wind_gust_amplitude,
            "wind_gust_period": config.wind_gust_period,
            "enable_drag": config.enable_drag,
            "drag_coefficient": config.drag_coefficient,
        }
    }


# ============================================================
# Phase 6: Kalman Filter & Sensor Fusion Endpoints
# ============================================================

# Global fusion manager (reset with each run)
fusion_manager: Optional[TrackFusionManager] = None


@app.get("/sensor/tracks")
async def get_sensor_tracks():
    """
    Get detailed track information including Kalman state.

    Returns tracks from each interceptor's sensor with uncertainty data.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state
    sensor_model = SensorModel()  # Use default config

    tracks_by_sensor = {}
    for interceptor in state.interceptors:
        sensor_state = sensor_states.get(interceptor.id)
        if sensor_state is None:
            continue

        sensor_tracks = []
        for track_id, track in sensor_state.tracks.items():
            track_data = {
                "track_id": track_id,
                "target_id": track.target_id,
                "position": track.estimated_position.to_dict(),
                "velocity": track.estimated_velocity.to_dict(),
                "track_quality": track.track_quality,
                "detections": track.detections,
                "coasting": track.coasting,
                "is_firm": track.is_firm,
                "position_uncertainty": track.get_position_uncertainty(),
            }

            # Include Kalman state details if available
            if track.kalman_state is not None:
                track_data["kalman"] = track.kalman_state.to_dict()

            sensor_tracks.append(track_data)

        tracks_by_sensor[interceptor.id] = sensor_tracks

    return {
        "timestamp": state.sim_time,
        "tracks_by_sensor": tracks_by_sensor,
        "total_tracks": sum(len(t) for t in tracks_by_sensor.values()),
    }


@app.get("/sensor/fused-tracks")
async def get_fused_tracks():
    """
    Get fused tracks combining data from multiple sensors.

    Phase 6: Multi-sensor fusion using Covariance Intersection.
    """
    global fusion_manager

    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state

    # Create fusion manager if needed
    if fusion_manager is None:
        fusion_manager = TrackFusionManager()

    # Add local tracks from each sensor to fusion manager
    for interceptor_id, sensor_state in sensor_states.items():
        for track_id, track in sensor_state.tracks.items():
            if track.kalman_state is not None and track.is_firm:
                fusion_manager.add_local_track(
                    sensor_id=interceptor_id,
                    track_id=f"{interceptor_id}_{track_id}",
                    target_id=track.target_id,
                    kalman_state=track.kalman_state,
                    timestamp=state.sim_time,
                    confidence=track.track_quality
                )

    # Fuse tracks
    fused_tracks = fusion_manager.fuse_associated_tracks(state.sim_time)

    return {
        "timestamp": state.sim_time,
        "fused_tracks": [ft.to_dict() for ft in fused_tracks],
        "num_fused_tracks": len(fused_tracks),
    }


class KalmanConfigRequest(BaseModel):
    """Request body for Kalman filter configuration."""
    process_noise_pos: float = 1.0
    process_noise_vel: float = 0.1
    measurement_noise_pos: float = 50.0
    initial_pos_variance: float = 100.0
    initial_vel_variance: float = 25.0


@app.post("/sensor/kalman/configure")
async def configure_kalman(config: KalmanConfigRequest):
    """
    Configure Kalman filter parameters.

    Note: Takes effect on next simulation run.
    """
    return {
        "status": "ok",
        "config": {
            "process_noise_pos": config.process_noise_pos,
            "process_noise_vel": config.process_noise_vel,
            "measurement_noise_pos": config.measurement_noise_pos,
            "initial_pos_variance": config.initial_pos_variance,
            "initial_vel_variance": config.initial_vel_variance,
        },
        "note": "Configuration will apply to next simulation run"
    }


# ============================================================
# Phase 6: Cooperative Engagement Endpoints
# ============================================================

@app.get("/cooperative/state")
async def get_cooperative_state():
    """
    Get current cooperative engagement state.

    Returns engagement zones, pending handoffs, and assignments.
    """
    if current_engine is None or current_engine.cooperative is None:
        return {
            "enabled": False,
            "engagement_zones": [],
            "pending_handoffs": [],
            "completed_handoffs": [],
            "interceptor_zones": {},
            "target_assignments": {},
        }

    coop = current_engine.cooperative
    state = coop.get_state()

    return {
        "enabled": True,
        **state.to_dict()
    }


@app.post("/cooperative/zones")
async def create_engagement_zone(request: EngagementZoneRequest):
    """
    Create a new engagement zone (killbox).

    Zones define sectors for cooperative defense.
    """
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(
            status_code=400,
            detail="No active simulation with cooperative engagement enabled"
        )

    coop = current_engine.cooperative
    from sim.vector import Vec3

    zone_id = coop.create_zone(
        name=request.name,
        center=Vec3(request.center_x, request.center_y, request.center_z),
        dimensions=Vec3(request.width, request.depth, request.height),
        rotation=request.rotation,
        priority=request.priority,
        color=request.color,
    )

    zone = coop.get_zone(zone_id)

    return {
        "status": "created",
        "zone": zone.to_dict()
    }


@app.get("/cooperative/zones")
async def list_engagement_zones():
    """List all engagement zones."""
    if current_engine is None or current_engine.cooperative is None:
        return {"zones": []}

    zones = current_engine.cooperative.get_zones()
    return {"zones": [z.to_dict() for z in zones]}


@app.get("/cooperative/zones/{zone_id}")
async def get_engagement_zone(zone_id: str):
    """Get details for a specific zone."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    zone = current_engine.cooperative.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")

    return zone.to_dict()


@app.delete("/cooperative/zones/{zone_id}")
async def delete_engagement_zone(zone_id: str):
    """Delete an engagement zone."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.cooperative.delete_zone(zone_id):
        return {"status": "deleted", "zone_id": zone_id}

    raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")


@app.post("/cooperative/zones/assign")
async def assign_interceptor_to_zone(assignment: InterceptorZoneAssignment):
    """Assign an interceptor to an engagement zone."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.cooperative.assign_interceptor_to_zone(
        assignment.interceptor_id,
        assignment.zone_id
    ):
        return {
            "status": "assigned",
            "interceptor_id": assignment.interceptor_id,
            "zone_id": assignment.zone_id
        }

    raise HTTPException(
        status_code=400,
        detail=f"Could not assign {assignment.interceptor_id} to {assignment.zone_id}"
    )


@app.post("/cooperative/zones/unassign/{interceptor_id}")
async def unassign_interceptor_from_zone(interceptor_id: str):
    """Remove an interceptor from its assigned zone."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.cooperative.unassign_interceptor(interceptor_id):
        return {"status": "unassigned", "interceptor_id": interceptor_id}

    raise HTTPException(
        status_code=400,
        detail=f"Interceptor {interceptor_id} is not assigned to any zone"
    )


@app.post("/cooperative/handoff/request")
async def request_handoff(request: HandoffRequestModel):
    """
    Request a target handoff between interceptors.

    Used for zone transitions or manual reassignment.
    """
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    # Map reason string to enum
    try:
        reason = HandoffReason(request.reason)
    except ValueError:
        reason = HandoffReason.MANUAL

    request_id = current_engine.cooperative.request_handoff(
        from_interceptor=request.from_interceptor,
        to_interceptor=request.to_interceptor,
        target_id=request.target_id,
        reason=reason,
        current_time=current_engine.state.sim_time
    )

    if request_id:
        return {
            "status": "requested",
            "request_id": request_id,
            "from": request.from_interceptor,
            "to": request.to_interceptor,
            "target": request.target_id
        }

    raise HTTPException(
        status_code=400,
        detail="Handoff request rejected (cooldown, not assigned, or pending handoff exists)"
    )


@app.post("/cooperative/handoff/approve/{request_id}")
async def approve_handoff(request_id: str):
    """Approve a pending handoff request."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    if current_engine.cooperative.approve_handoff(
        request_id,
        current_engine.state.sim_time
    ):
        return {"status": "approved", "request_id": request_id}

    raise HTTPException(status_code=400, detail=f"Could not approve handoff {request_id}")


@app.post("/cooperative/handoff/reject/{request_id}")
async def reject_handoff(request_id: str):
    """Reject a pending handoff request."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.cooperative.reject_handoff(request_id):
        return {"status": "rejected", "request_id": request_id}

    raise HTTPException(status_code=400, detail=f"Could not reject handoff {request_id}")


@app.post("/cooperative/handoff/execute/{request_id}")
async def execute_handoff(request_id: str):
    """Execute an approved handoff."""
    if current_engine is None or current_engine.cooperative is None:
        raise HTTPException(status_code=400, detail="Cooperative not enabled")

    if current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    if current_engine.cooperative.execute_handoff(
        request_id,
        current_engine.state.sim_time
    ):
        return {"status": "executed", "request_id": request_id}

    raise HTTPException(
        status_code=400,
        detail=f"Could not execute handoff {request_id} (not approved?)"
    )


@app.get("/cooperative/handoffs")
async def list_handoffs():
    """List pending and recent handoff requests."""
    if current_engine is None or current_engine.cooperative is None:
        return {
            "pending": [],
            "approved": [],
            "completed": []
        }

    coop = current_engine.cooperative
    return {
        "pending": [h.to_dict() for h in coop.get_pending_handoffs()],
        "approved": [h.to_dict() for h in coop.get_approved_handoffs()],
        "completed": [h.to_dict() for h in coop.state.completed_handoffs[-10:]]
    }


# ============================================================
# Phase 6.4: ML/AI Endpoints
# ============================================================

class MLModelLoadRequest(BaseModel):
    """Request body for loading an ML model."""
    model_id: str
    model_path: str
    model_type: str = "threat_assessment"  # "threat_assessment" or "guidance"
    device: str = "cpu"
    num_threads: int = 4


class MLModelActivateRequest(BaseModel):
    """Request body for activating a model."""
    model_id: str
    model_type: str = "threat_assessment"  # "threat_assessment" or "guidance"


@app.get("/ml/status")
async def get_ml_status():
    """
    Get ML subsystem status.

    Returns whether ONNX runtime is available and loaded models.
    """
    registry = get_model_registry()

    return {
        "onnx_available": ONNX_AVAILABLE,
        "models": registry.list_models(),
        "active_threat_model": registry.active_threat_model,
        "active_guidance_model": registry.active_guidance_model,
    }


@app.get("/ml/models")
async def list_ml_models():
    """
    List all loaded ML models.

    Returns model IDs, types, and status.
    """
    registry = get_model_registry()
    return registry.list_models()


@app.post("/ml/models/load")
async def load_ml_model(request: MLModelLoadRequest):
    """
    Load an ML model from file.

    Supports both threat assessment and guidance models in ONNX format.
    """
    if not ONNX_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="ONNX runtime not installed. Install with: pip install onnxruntime"
        )

    registry = get_model_registry()
    config = MLConfig(
        model_path=request.model_path,
        model_type=request.model_type,
        device=request.device,
        num_threads=request.num_threads,
    )

    if request.model_type == "threat_assessment":
        success = registry.load_threat_model(request.model_id, request.model_path, config)
    elif request.model_type == "guidance":
        success = registry.load_guidance_model(request.model_id, request.model_path, config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")

    if success:
        return {
            "status": "loaded",
            "model_id": request.model_id,
            "model_type": request.model_type,
            "path": request.model_path
        }

    raise HTTPException(status_code=400, detail=f"Failed to load model from {request.model_path}")


@app.delete("/ml/models/{model_type}/{model_id}")
async def unload_ml_model(model_type: str, model_id: str):
    """
    Unload an ML model.

    Args:
        model_type: "threat_assessment" or "guidance"
        model_id: ID of the model to unload
    """
    registry = get_model_registry()

    if model_type == "threat_assessment":
        success = registry.unload_threat_model(model_id)
    elif model_type == "guidance":
        success = registry.unload_guidance_model(model_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    if success:
        return {"status": "unloaded", "model_id": model_id}

    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@app.post("/ml/models/activate")
async def activate_ml_model(request: MLModelActivateRequest):
    """
    Set a model as the active model for its type.

    The active model is used for predictions during simulation.
    """
    registry = get_model_registry()

    if request.model_type == "threat_assessment":
        if request.model_id not in registry.threat_models:
            raise HTTPException(status_code=404, detail=f"Threat model {request.model_id} not found")
        registry.active_threat_model = request.model_id
    elif request.model_type == "guidance":
        if request.model_id not in registry.guidance_models:
            raise HTTPException(status_code=404, detail=f"Guidance model {request.model_id} not found")
        registry.active_guidance_model = request.model_id
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")

    return {
        "status": "activated",
        "model_id": request.model_id,
        "model_type": request.model_type
    }


@app.get("/ml/threat-assessment")
async def get_ml_threat_assessment(mode: str = "ml"):
    """
    Get ML-based threat assessment for current simulation.

    Args:
        mode: "ml" (ML only), "rule" (rule-based only), or "hybrid" (blend)

    Returns threat scores from the ML model or fallback if not available.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state
    registry = get_model_registry()
    model = registry.get_active_threat_model()

    assessments = []
    for interceptor in state.interceptors:
        if mode == "ml":
            if model is None or not model.model_loaded:
                # Create fallback model (uses heuristic)
                model = ThreatModel()

            geometries = [
                compute_intercept_geometry(interceptor, target)
                for target in state.targets
            ]
            assessment = ml_threat_assessment(interceptor, state.targets, model, geometries)
        elif mode == "hybrid":
            geometries = [
                compute_intercept_geometry(interceptor, target)
                for target in state.targets
            ]
            assessment = hybrid_threat_assessment(
                interceptor, state.targets, model,
                ml_weight=0.5 if model and model.model_loaded else 0.0
            )
        else:  # rule
            geometries = [
                compute_intercept_geometry(interceptor, target)
                for target in state.targets
            ]
            assessment = assess_all_threats(interceptor, state.targets, geometries)

        assessments.append(assessment.to_dict())

    return {
        "mode": mode,
        "model_active": model is not None and model.model_loaded if model else False,
        "assessments": assessments
    }


@app.get("/ml/features/{interceptor_id}/{target_id}")
async def get_ml_features(interceptor_id: str, target_id: str):
    """
    Get extracted ML features for a specific interceptor-target pair.

    Useful for debugging and understanding model inputs.
    """
    if current_engine is None or current_engine.state is None:
        raise HTTPException(status_code=400, detail="No active simulation")

    state = current_engine.state

    # Find interceptor
    interceptor = next((i for i in state.interceptors if i.id == interceptor_id), None)
    if not interceptor:
        raise HTTPException(status_code=404, detail=f"Interceptor {interceptor_id} not found")

    # Find target
    target = next((t for t in state.targets if t.id == target_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Target {target_id} not found")

    # Extract features
    from sim.ml.features import extract_threat_features, extract_guidance_features, ThreatFeatures, GuidanceFeatures
    from sim.intercept import compute_intercept_geometry

    geometry = compute_intercept_geometry(interceptor, target)
    threat_features = extract_threat_features(interceptor, target, geometry)
    guidance_features = extract_guidance_features(interceptor, target, geometry)

    return {
        "interceptor_id": interceptor_id,
        "target_id": target_id,
        "threat_features": {
            "values": threat_features.to_numpy().tolist(),
            "names": ThreatFeatures.feature_names(),
        },
        "guidance_features": {
            "values": guidance_features.to_numpy().tolist(),
            "names": GuidanceFeatures.feature_names(),
        }
    }


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time event streaming.

    The UI connects here to receive:
    - State updates (50 Hz during run)
    - Completion events
    - Error events

    This is the "data bus" of the system.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
            # Client can send commands (future: manual control)
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
