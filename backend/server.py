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
        name: {"name": name, "description": info["description"]}
        for name, info in SCENARIOS.items()
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

    # Create engine with config
    sim_config = SimConfig(
        dt=config.dt,
        max_time=config.max_time,
        kill_radius=config.kill_radius,
        real_time=config.real_time,
    )
    current_engine = SimEngine(config=sim_config)

    # Register broadcast handler
    current_engine.on_event(manager.broadcast)

    # Setup scenario
    current_engine.setup_scenario(
        target_start=scenario["target_start"],
        target_velocity=scenario["target_velocity"],
        interceptor_start=scenario["interceptor_start"],
        interceptor_velocity=scenario["interceptor_velocity"],
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
