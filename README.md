# Air Dominance Simulation

A real-time air intercept simulation sandbox for learning guidance, control, and autonomy concepts.

## Phase 3: Evasion, Envelopes & Multi-Interceptor ✓

Building on Phase 2, this phase adds advanced tactical features:

- **Target Evasion Maneuvers**: Test guidance against maneuvering targets
  - Constant Turn: Sustained turn at fixed rate
  - Weave (S-Turns): Periodic direction reversals
  - Barrel Roll: 3D spiral evasion maneuver
  - Random Jink: Unpredictable random direction changes
- **Engagement Envelope Analysis**: Find the weapon system's reach
  - Sweep across range, bearing, and elevation
  - Monte Carlo at each point for statistical confidence
  - 2D Heatmap visualization (range × bearing)
  - Identify intercept probability boundaries
- **Multiple Interceptors**: Spawn 1-8 interceptors with formation spacing
  - Each runs independent guidance
  - Distinct color-coded visualization
  - First-to-intercept wins
- **Enhanced 3D Trails**: Gradient trails with time markers
- **Evasive Scenario Presets**: Pre-configured scenarios with evasion

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
│  ┌─────────────┐  ┌──────────────┐            │             │
│  │ Three.js    │  │ Heatmap/     │            │             │
│  │ 3D Scene    │  │ Charts       │            │             │
│  └─────────────┘  └──────────────┘            │             │
└───────────────────────────────────────────────│─────────────┘
                                                │ WebSocket
                                                │ (50 Hz)
┌───────────────────────────────────────────────│─────────────┐
│                        Backend                │             │
│  ┌────────────────┐  ┌────────────┐  ┌───────▼──────────┐  │
│  │ SimEngine      │──│ Guidance   │  │ FastAPI Server   │  │
│  │ - Entity mgmt  │  │ - PP/PN    │  │ - REST API       │  │
│  │ - Physics      │  │ - APN      │  │ - WebSocket      │  │
│  │ - Multi-intcpt │  └────────────┘  │   broadcast      │  │
│  └────────────────┘         │        └──────────────────┘  │
│         │            ┌──────▼─────┐                        │
│  ┌──────▼─────┐      │ Envelope   │                        │
│  │ Evasion    │      │ Analysis   │                        │
│  │ - Turn     │      │ - Sweep    │                        │
│  │ - Weave    │      │ - Heatmap  │                        │
│  │ - Barrel   │      └────────────┘                        │
│  │ - Jink     │                                            │
│  └────────────┘                                            │
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

### Target Evasion Maneuvers

Test guidance against maneuvering targets:

**Constant Turn**: Sustained horizontal turn at fixed rate. Creates centripetal acceleration perpendicular to velocity. Simple but effective against pure pursuit.

**Weave (S-Turns)**: Sinusoidal turn rate creates periodic direction reversals. Degrades PN guidance accuracy by creating unpredictable LOS rate changes.

**Barrel Roll**: 3D spiral combining horizontal turn and vertical oscillation. Creates a corkscrew-like path that challenges guidance systems in all axes.

**Random Jink**: Unpredictable direction changes at random intervals. Maximizes uncertainty - most difficult to counter.

### Engagement Envelope

The engagement envelope defines where intercepts are possible:
- Sweep range (1-5km), bearing (-90° to +90°), and elevation
- Run Monte Carlo at each point for statistical confidence
- Visualize as 2D heatmap (range × bearing)
- Find the boundary of effective weapon reach

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
│   │   ├── engine.py       # Simulation loop, physics, multi-interceptor
│   │   ├── guidance.py     # Guidance laws (PP, PN, APN)
│   │   ├── evasion.py      # Target evasion maneuvers
│   │   ├── envelope.py     # Engagement envelope analysis
│   │   └── monte_carlo.py  # Batch simulation & stats
│   └── server.py           # FastAPI + WebSocket server
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx        # Three.js 3D visualization + trails
│       │   └── ControlPanel.tsx # UI controls, evasion, envelope
│       ├── hooks/
│       │   └── useSimulation.ts # WebSocket + API state management
│       ├── types.ts             # TypeScript interfaces
│       └── App.tsx
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scenarios` | GET | List preset scenarios (including evasive) |
| `/runs` | POST | Start a simulation run |
| `/runs/current` | GET | Get current run status |
| `/runs/stop` | POST | Stop the current run |
| `/guidance` | GET | List available guidance laws |
| `/evasion` | GET | List available evasion maneuvers |
| `/monte-carlo` | POST | Run Monte Carlo batch analysis |
| `/monte-carlo/sweep` | POST | Parameter sweep analysis |
| `/envelope` | POST | Compute engagement envelope |
| `/ws` | WebSocket | Real-time 50Hz state stream |

## Next: Phase 4

- **Optimal Intercept Geometry**: Compute lead-pursuit angles and optimal approach vectors
- **Threat Assessment**: Score targets by approach angle, speed, and time-to-impact
- **Weapon-Target Assignment (WTA)**: Multi-interceptor coordination and resource allocation
- **Sensor Modeling**: Radar detection ranges, look angles, and track quality
- **Recording & Replay**: Save engagements and replay with different parameters
