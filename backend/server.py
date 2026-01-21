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

from sim.engine import SimEngine, SimConfig, load_scenario, SCENARIOS
from sim.vector import Vec3
from sim.guidance import GuidanceType, GuidanceParams, create_guidance_function
from sim.monte_carlo import MonteCarloConfig, run_monte_carlo, parameter_sweep
from sim.evasion import EvasionType, EvasionConfig
from sim.envelope import EnvelopeConfig, compute_engagement_envelope


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


# ============================================================
# API Models
# ============================================================

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

    # Create engine with config
    sim_config = SimConfig(
        dt=config.dt,
        max_time=config.max_time,
        kill_radius=config.kill_radius,
        real_time=config.real_time,
    )
    current_engine = SimEngine(
        config=sim_config,
        guidance=guidance,
        evasion_type=evasion_type,
    )

    # Register broadcast handler
    current_engine.on_event(manager.broadcast)

    # Setup scenario
    current_engine.setup_scenario(
        target_start=scenario["target_start"],
        target_velocity=scenario["target_velocity"],
        interceptor_start=scenario["interceptor_start"],
        interceptor_velocity=scenario["interceptor_velocity"],
        num_interceptors=config.num_interceptors,
    )

    # Start simulation in background
    run_task = asyncio.create_task(current_engine.run())

    return {
        "run_id": current_engine.state.run_id,
        "scenario": config.scenario,
        "status": "started",
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
