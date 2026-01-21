# Air Dominance Simulation

A real-time air intercept simulation sandbox for learning guidance, control, and autonomy concepts.

## Phase 1: Foundation

This phase implements:
- **3D Kinematics**: Position, velocity, acceleration in 3D space
- **Fixed Timestep Simulation**: Deterministic 50Hz physics
- **Pure Pursuit Guidance**: Simple "point at target" guidance law
- **Real-time WebSocket Streaming**: 50Hz state updates to UI
- **Three.js 3D Visualization**: Watch the intercept in real-time

## Quick Start

### 1. Start the Backend (Python)

```bash
cd backend
uv sync  # or: pip install -e .
python server.py
```

Backend runs at: http://localhost:8000

### 2. Start the Frontend (React + Three.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

### 3. Run a Scenario

1. Open http://localhost:5173 in your browser
2. Click one of the scenario buttons (HEAD ON, CROSSING, TAIL CHASE)
3. Watch the interceptor (blue) chase the target (red)
4. Use mouse to rotate/zoom the 3D view

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ React App   │──│ useSimulation│──│ WebSocket Client   │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│         │                                      │             │
│  ┌─────────────┐                              │             │
│  │ Three.js    │                              │             │
│  │ 3D Scene    │                              │             │
│  └─────────────┘                              │             │
└───────────────────────────────────────────────│─────────────┘
                                                │ WebSocket
                                                │ (50 Hz)
┌───────────────────────────────────────────────│─────────────┐
│                        Backend                │             │
│  ┌────────────────┐  ┌────────────┐  ┌───────▼──────────┐  │
│  │ SimEngine      │──│ Guidance   │  │ FastAPI Server   │  │
│  │ - Entity mgmt  │  │ - Pure     │  │ - REST API       │  │
│  │ - Physics      │  │   Pursuit  │  │ - WebSocket      │  │
│  │ - End detect   │  │            │  │   broadcast      │  │
│  └────────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Guidance Laws

**Pure Pursuit** (current implementation):
- Points directly at target's current position
- Simple but inefficient (curved path)
- Can't hit fast-maneuvering targets well

**Coming in Phase 2: Proportional Navigation**
- Predicts intercept point
- More efficient (straighter path)
- Industry standard for missiles

### Coordinate System

We use ENU (East-North-Up):
- X: East (positive = right)
- Y: North (positive = forward)
- Z: Up (positive = above ground)

Units:
- Position: meters
- Velocity: m/s
- Acceleration: m/s²
- Time: seconds

## Project Structure

```
air-dominance/
├── backend/
│   ├── sim/
│   │   ├── vector.py      # 3D vector math
│   │   ├── entities.py    # Target & Interceptor models
│   │   └── engine.py      # Simulation loop & guidance
│   └── server.py          # FastAPI + WebSocket server
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx       # Three.js 3D visualization
│       │   └── ControlPanel.tsx # UI controls & telemetry
│       ├── hooks/
│       │   └── useSimulation.ts # WebSocket state management
│       └── App.tsx
└── README.md
```

## Next: Phase 2

- Proportional Navigation guidance
- Monte Carlo parameter sweeps
- Engagement envelope analysis
- Multiple interceptors
