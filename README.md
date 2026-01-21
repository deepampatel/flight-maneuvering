# Air Dominance Simulation

A real-time air intercept simulation sandbox for learning guidance, control, and autonomy concepts.

## Phase 5: Multi-Target, Sensors & Weapon-Target Assignment ✓

Building on Phase 4, this phase adds multi-target engagement capabilities:

- **Multi-Target Support**: Engage multiple simultaneous threats
  - 1-4 targets per scenario
  - Each target tracked independently
  - Intercepts tracked per target-interceptor pair
  - Pre-built multi-target scenarios (2, 3, 4 targets)
- **Sensor Modeling**: Realistic detection and tracking
  - Range-limited detection (configurable max range)
  - Field-of-view constraints (120° default)
  - Detection probability based on range and aspect
  - Measurement noise (range and angle)
  - Track quality and confidence scoring
- **Weapon-Target Assignment (WTA)**: Automated interceptor allocation
  - Greedy Nearest: Each interceptor takes closest target
  - Greedy Threat: Prioritize highest-threat targets
  - Hungarian (Optimal): Globally optimal assignment
  - Real-time reassignment as situation evolves
  - Assignment visualization in HUD
- **Enhanced UI**: Multi-target controls and status
  - Target count slider (1-4)
  - WTA algorithm selector
  - Kill tracking panel
  - Assignment display

## Phase 4: Intercept Geometry, Threat Assessment & Recording ✓

Building on Phase 3, this phase adds tactical decision support and mission replay:

- **Intercept Geometry Analysis**: Real-time computation of engagement parameters
  - Line-of-Sight (LOS) range and rate
  - Aspect angle and Antenna Train Angle (ATA)
  - Lead angle for optimal intercept
  - Time-to-intercept (TTI) estimation
  - Closing velocity and collision course detection
- **Threat Assessment**: Score and prioritize targets
  - Time-to-impact scoring (closer = higher threat)
  - Closing velocity analysis
  - Aspect angle evaluation
  - Threat levels: CRITICAL, HIGH, MEDIUM, LOW
  - Engagement recommendations
- **Recording & Replay System**: Capture and analyze engagements
  - Record simulation state at 50Hz
  - Save to disk with metadata
  - Replay with pause/resume controls
  - Delete old recordings
- **Mission Control UI Redesign**: Compact, aerospace-style interface
  - Horizontal toolbar with all controls
  - Full-screen 3D visualization
  - Floating telemetry HUD overlay
  - Slide-out advanced panel (Monte Carlo, Envelope, Recordings)

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
2. Select scenario, guidance law, evasion type from the toolbar
3. Click LAUNCH to start the simulation
4. Watch the interceptor (blue) chase the target (red)
5. Use mouse to rotate/zoom the 3D view
6. Click ADV to access Monte Carlo, Envelope, and Recordings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ React App   │──│ useSimulation│──│ WebSocket Client   │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
│         │                                      │             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────▼──────────┐  │
│  │ Three.js    │  │ Telemetry    │  │ Recording/Replay  │  │
│  │ 3D Scene    │  │ HUD          │  │ Controls          │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
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
│         │            ┌──────▼─────┐         │              │
│  ┌──────▼─────┐      │ Intercept  │  ┌──────▼──────────┐  │
│  │ Evasion    │      │ Geometry   │  │ Recording       │  │
│  │ - Turn     │      │ - LOS/TTI  │  │ - Capture       │  │
│  │ - Weave    │      │ - Aspect   │  │ - Playback      │  │
│  │ - Barrel   │      └────────────┘  │ - Storage       │  │
│  │ - Jink     │             │        └─────────────────┘  │
│  └────────────┘      ┌──────▼─────┐                       │
│                      │ Threat     │                       │
│                      │ Assessment │                       │
│                      │ - Scoring  │                       │
│                      │ - Priority │                       │
│                      └────────────┘                       │
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

### Intercept Geometry

Real-time computation of engagement parameters:

**Line-of-Sight (LOS)**: Vector from interceptor to target
- LOS Range: Distance to target
- LOS Rate: Rate of change of LOS angle (critical for PN)

**Aspect Angle**: Angle between target velocity and LOS
- 0° = head-on approach
- 180° = tail chase

**Antenna Train Angle (ATA)**: Angle from interceptor nose to LOS
- Determines sensor look angle requirements

**Time-to-Intercept (TTI)**: Estimated time until closest approach
- Computed from closing velocity and range

### Threat Assessment

Score and prioritize multiple targets:

**Scoring Factors**:
- Time-to-impact: Closer = higher threat
- Closing velocity: Faster approach = higher threat
- Aspect angle: Head-on = higher threat

**Threat Levels**:
- CRITICAL: Score > 80 (immediate action required)
- HIGH: Score > 60
- MEDIUM: Score > 40
- LOW: Score ≤ 40

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
│   │   ├── engine.py       # Simulation loop, physics, multi-target/interceptor
│   │   ├── guidance.py     # Guidance laws (PP, PN, APN)
│   │   ├── evasion.py      # Target evasion maneuvers
│   │   ├── intercept.py    # Intercept geometry calculations
│   │   ├── threat.py       # Threat assessment and scoring
│   │   ├── recording.py    # Simulation recording and replay
│   │   ├── envelope.py     # Engagement envelope analysis
│   │   ├── monte_carlo.py  # Batch simulation & stats
│   │   ├── sensor.py       # Sensor modeling (Phase 5)
│   │   └── assignment.py   # Weapon-Target Assignment (Phase 5)
│   ├── recordings/         # Saved simulation recordings
│   └── server.py           # FastAPI + WebSocket server
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx        # Three.js 3D visualization + trails
│       │   └── ControlPanel.tsx # Mission control toolbar + HUD
│       ├── hooks/
│       │   └── useSimulation.ts # WebSocket + API state management
│       ├── types.ts             # TypeScript interfaces (Phase 5: WTA, Sensor types)
│       └── App.tsx
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scenarios` | GET | List preset scenarios (including multi-target) |
| `/runs` | POST | Start a simulation run |
| `/runs/current` | GET | Get current run status |
| `/runs/stop` | POST | Stop the current run |
| `/guidance` | GET | List available guidance laws |
| `/evasion` | GET | List available evasion maneuvers |
| `/intercept-geometry` | GET | Get current intercept geometry |
| `/threat-assessment` | GET | Get current threat assessment |
| `/sensor/config` | GET | Get sensor configuration |
| `/sensor/detections` | GET | Get current sensor detections |
| `/wta/algorithms` | GET | List WTA algorithms |
| `/wta/assignments` | GET | Get current WTA assignments |
| `/wta/cost-matrix` | GET | Get assignment cost matrix |
| `/recordings` | GET | List saved recordings |
| `/recordings` | POST | Start recording |
| `/recordings/stop` | POST | Stop recording |
| `/recordings/{id}` | DELETE | Delete a recording |
| `/recordings/{id}/replay` | POST | Start replay |
| `/replay/pause` | POST | Pause replay |
| `/replay/resume` | POST | Resume replay |
| `/replay/stop` | POST | Stop replay |
| `/monte-carlo` | POST | Run Monte Carlo batch analysis |
| `/monte-carlo/sweep` | POST | Parameter sweep analysis |
| `/envelope` | POST | Compute engagement envelope |
| `/ws` | WebSocket | Real-time 50Hz state stream |

## Coming Up: Phase 6

- **Environmental Effects**: Atmospheric and terrain factors
  - Wind effects on trajectories
  - Terrain masking and line-of-sight blockage
  - Weather effects on sensor performance
- **AI/ML Integration**: Learning-based guidance and decision making
  - Reinforcement learning for adaptive guidance
  - Neural network threat assessment
  - Learned evasion patterns
- **Advanced Sensor Fusion**: Combine multiple sensor inputs
  - Track-to-track fusion
  - Data association
  - Kalman filtering for state estimation
- **Cooperative Engagement**: Multi-platform coordination
  - Distributed fire control
  - Hand-off between platforms
  - Engagement zones
