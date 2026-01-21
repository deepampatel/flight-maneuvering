# Air Dominance Simulation

A real-time air intercept simulation sandbox for learning guidance, control, and autonomy concepts.

## Phase 2: Guidance Laws & Monte Carlo Analysis ✓

Building on Phase 1, this phase adds:
- **Proportional Navigation (PN)**: Industry-standard missile guidance law
- **Augmented PN**: Better performance against maneuvering targets
- **Selectable Guidance Laws**: Switch between Pure Pursuit, PN, and Augmented PN
- **Tunable Navigation Constant**: Adjust N parameter (1-8) for PN laws
- **Monte Carlo Simulation**: Run 100+ simulations with noise to test robustness
- **Statistical Analysis**: Intercept rate, miss distance distribution, histograms

## Phase 1: Foundation ✓

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

**Pure Pursuit**:
- Points directly at target's current position
- Simple but inefficient (curved path)
- Can't hit fast-maneuvering targets well

**Proportional Navigation (PN)**:
- Commands turn rate proportional to line-of-sight (LOS) rate
- Drives LOS rate to zero → constant bearing → collision course
- Formula: `a = N × Vc × LOS_rate` (N typically 3-5)
- Industry standard for missiles

**Augmented PN**:
- Adds target acceleration compensation
- Better against maneuvering targets
- Formula: `a = N × Vc × LOS_rate + (N/2) × At`

### Monte Carlo Analysis

Test guidance robustness under uncertainty:
- Run 100+ simulations with random variations
- Add Gaussian noise to initial positions (50m std dev)
- Add noise to velocities (5 m/s std dev)
- View intercept rate, miss distance statistics, histograms

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
│   │   ├── vector.py       # 3D vector math
│   │   ├── entities.py     # Target & Interceptor models
│   │   ├── engine.py       # Simulation loop & physics
│   │   ├── guidance.py     # Guidance laws (PP, PN, APN)
│   │   └── monte_carlo.py  # Batch simulation & stats
│   └── server.py           # FastAPI + WebSocket server
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx        # Three.js 3D visualization
│       │   └── ControlPanel.tsx # UI controls, guidance, Monte Carlo
│       ├── hooks/
│       │   └── useSimulation.ts # WebSocket + API state management
│       └── App.tsx
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scenarios` | GET | List preset scenarios |
| `/scenarios/{name}` | POST | Start a preset scenario |
| `/guidance` | GET | List available guidance laws |
| `/monte-carlo` | POST | Run Monte Carlo batch analysis |
| `/monte-carlo/sweep` | POST | Parameter sweep analysis |
| `/ws` | WebSocket | Real-time 50Hz state stream |

## Next: Phase 3

- Engagement envelope analysis
- Multiple interceptors
- Target evasion maneuvers
- 3D trajectory trails in UI
