# Air Dominance Simulation

A real-time air intercept simulation sandbox for learning guidance, control, and autonomy concepts.

## Phase 8: Launch Platforms & Command Center UX ✓

Building on Phase 7, this phase adds mobile launch platforms (bogeys) and command center visualization:

- **Launch Platforms (Bogeys)**: Mobile missile launch systems
  - Autonomous launch platforms with configurable detection range
  - Magazine management (1-20 missiles per launcher)
  - Auto-launch mode with target acquisition
  - Lead prediction for optimal launch direction
  - Detection range visualization rings
  - Mission planner integration for placement
- **Command Center Alerts**: Real-time event notification system
  - Target acquisition alerts ("TGT ACQUIRED")
  - Missile launch alerts ("FOX THREE")
  - Magazine depletion warnings ("WINCHESTER")
  - Track lost notifications
  - Military-style callouts with visual effects
  - Auto-dismissing toast notifications with progress bars
- **Enhanced Guidance**: Improved proportional navigation
  - Pursuit component added to PN for better closing
  - Remaining acceleration budget utilized for LOS tracking
  - Fixed HMT approval integration for seamless guidance
- **WebSocket Event Streaming**: Real-time launcher state
  - Tracked targets per launcher
  - Missile counts and engaged targets
  - Launch event detection in frontend

## Phase 7: Swarm Tactics, Terrain, Communications & Human-Machine Teaming ✓

Building on Phase 6, this phase adds four major feature sets for advanced multi-agent scenarios:

- **Swarm Tactics & Coordination**: Multi-agent autonomous behavior
  - Formation control (V-formation, line abreast, echelon, wedge, trail, diamond)
  - Reynolds flocking (separation, alignment, cohesion)
  - Leader following with dynamic slot assignment
  - Collision avoidance
  - Tactical maneuvers (saturation, pincer, defensive sphere)
- **Advanced Terrain & 3D Environment**: Physical world modeling
  - Digital Elevation Model (DEM) integration
  - Procedural terrain generation (multi-octave noise)
  - Line-of-sight terrain masking
  - Radar horizon calculations (4/3 Earth radius)
  - Urban environment (building LOS blocking)
- **Communication & Datalink Modeling**: Information flow constraints
  - Bandwidth-limited transmission (configurable kbps)
  - Latency with jitter modeling
  - Packet loss (range-dependent + random)
  - Priority queue for critical messages
  - Jamming effects
  - Networked fire control coordination
- **Human-Machine Teaming**: Operator interaction patterns
  - Authority levels (full auto, human-on-loop, human-in-loop, manual)
  - Action proposal and approval workflow
  - Workload metrics (decisions/minute, fatigue level)
  - Trust calibration (AI accuracy, override rate)
  - Timeout handling with configurable auto-approve

## Phase 6: Environmental Effects, Kalman Filter, Cooperative Engagement & AI/ML ✓

Building on Phase 5, this phase adds four major feature sets:

- **Environmental Effects**: Atmospheric realism
  - Wind velocity with configurable gusts
  - Altitude-dependent air density (barometric formula)
  - Aerodynamic drag (optional, disabled by default)
  - Wind indicator visualization
- **Kalman Filter & Sensor Fusion**: Proper state estimation
  - 6-state Kalman filter (position + velocity)
  - Process and measurement noise modeling
  - Multi-sensor track fusion
  - Uncertainty ellipsoid visualization
  - Track confidence scoring
- **Cooperative Engagement**: Multi-platform coordination
  - Engagement zones (killboxes)
  - Interceptor handoffs between zones
  - Shared track management
  - Handoff arc visualization
- **AI/ML Integration**: Neural network threat assessment
  - ONNX runtime for model inference
  - 18-feature threat model input
  - 24-feature guidance policy input
  - Hybrid rule-based + ML scoring
  - Model registry and hot-loading

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
│         │            │ Threat     │  ┌─────────────────┐  │
│  ┌──────▼─────┐      │ Assessment │  │ Environment     │  │
│  │ Sensor     │      │ - Scoring  │  │ - Wind/Gusts    │  │
│  │ - Detect   │      │ - Priority │  │ - Drag          │  │
│  │ - Track    │      │ - ML Model │  │ - Density       │  │
│  │ - Kalman   │      └────────────┘  └─────────────────┘  │
│  └────────────┘             │                             │
│         │            ┌──────▼─────┐  ┌─────────────────┐  │
│  ┌──────▼─────┐      │ Cooperative│  │ ML Module       │  │
│  │ Fusion     │      │ Engagement │  │ - ONNX Infer    │  │
│  │ - Track    │      │ - Zones    │  │ - Features      │  │
│  │ - Assoc    │      │ - Handoffs │  │ - Threat/Guide  │  │
│  └────────────┘      └────────────┘  └─────────────────┘  │
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

### Environmental Effects (Phase 6)

**Wind Model**: Configurable wind field with gusts
- Base wind velocity (direction + speed)
- Sinusoidal gust amplitude and period
- Affects relative velocity for drag calculations

**Atmospheric Density**: Barometric formula for altitude variation
- ρ(h) = ρ₀ × exp(-h / H)
- Sea level density: 1.225 kg/m³
- Scale height: 8500m

**Aerodynamic Drag**: Optional drag force modeling
- F_d = 0.5 × ρ × v² × C_d × A
- Configurable drag coefficient per entity
- Cross-sectional area parameter

### Kalman Filter (Phase 6)

6-state Extended Kalman Filter for track estimation:
- State: [px, py, pz, vx, vy, vz]
- Process noise for position and velocity
- Measurement noise for range and angle
- Covariance matrix for uncertainty quantification
- Visualized as 3D uncertainty ellipsoids

### Cooperative Engagement (Phase 6)

Multi-platform coordination features:
- **Engagement Zones**: 3D killboxes with assigned interceptors
- **Handoffs**: Transfer target tracking between interceptors
- **Shared Tracks**: Fused track data across platforms

### AI/ML Integration (Phase 6)

Neural network models for decision support:
- **Threat Model**: 18-feature input, threat score output
- **Guidance Model**: 24-feature input, acceleration command output
- ONNX runtime for portable inference
- Hybrid mode combines rule-based + ML scoring

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
│   │   ├── sensor.py       # Sensor modeling with Kalman filter
│   │   ├── assignment.py   # Weapon-Target Assignment (WTA)
│   │   ├── environment.py  # Wind, drag, atmospheric effects (Phase 6)
│   │   ├── kalman.py       # Kalman filter implementation (Phase 6)
│   │   ├── fusion.py       # Multi-sensor track fusion (Phase 6)
│   │   ├── cooperation.py  # Cooperative engagement (Phase 6)
│   │   ├── ml/             # AI/ML module (Phase 6)
│   │   │   ├── __init__.py
│   │   │   ├── inference.py  # ONNX model loading/inference
│   │   │   └── features.py   # Feature extraction for ML
│   │   ├── swarm.py        # Swarm tactics & formations (Phase 7)
│   │   ├── terrain.py      # DEM & terrain masking (Phase 7)
│   │   ├── datalink.py     # Communication modeling (Phase 7)
│   │   ├── hmt.py          # Human-machine teaming (Phase 7)
│   │   └── launcher.py     # Launch platforms/bogeys (Phase 8)
│   ├── recordings/         # Saved simulation recordings
│   └── server.py           # FastAPI + WebSocket server
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx        # Three.js 3D visualization + trails
│       │   ├── ControlPanel.tsx # Mission control toolbar + HUD
│       │   ├── MissionPlanner.tsx # Interactive entity placement
│       │   ├── HMTToast.tsx       # Human-Machine Teaming approval UI
│       │   └── LaunchEventToast.tsx # Command center event alerts
│       ├── hooks/
│       │   └── useSimulation.ts # WebSocket + API state management
│       ├── types.ts             # TypeScript interfaces
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
| `/environment/config` | GET | Get environment configuration |
| `/environment/configure` | POST | Configure wind, drag, atmosphere |
| `/sensor/tracks` | GET | Get Kalman-filtered tracks |
| `/sensor/fused-tracks` | GET | Get multi-sensor fused tracks |
| `/sensor/kalman/configure` | POST | Configure Kalman filter |
| `/cooperative/state` | GET | Get cooperative engagement state |
| `/cooperative/zones` | POST | Create engagement zone |
| `/cooperative/zones/{id}` | DELETE | Delete engagement zone |
| `/cooperative/handoff/request` | POST | Request interceptor handoff |
| `/cooperative/handoff/execute/{id}` | POST | Execute pending handoff |
| `/ml/models` | GET | List available ML models |
| `/ml/models/load` | POST | Load ML model |
| `/ml/models/unload/{id}` | POST | Unload ML model |
| `/swarm/status` | GET | Get swarm subsystem status |
| `/swarm/formations` | GET | List available formations |
| `/swarm/configure` | POST | Configure swarm behavior |
| `/swarm/set-leader/{id}` | POST | Set swarm leader |
| `/terrain/status` | GET | Get terrain subsystem status |
| `/terrain/elevation` | GET | Get terrain elevation at point |
| `/terrain/los` | GET | Check line-of-sight between points |
| `/terrain/heightmap` | GET | Get terrain heightmap for visualization |
| `/terrain/generate` | POST | Generate procedural terrain |
| `/datalink/status` | GET | Get datalink subsystem status |
| `/datalink/configure` | POST | Configure datalink parameters |
| `/datalink/link-quality/{e1}/{e2}` | GET | Get link quality between entities |
| `/datalink/jammers` | POST | Add a jamming source |
| `/datalink/jammers/{id}` | DELETE | Remove a jamming source |
| `/hmt/status` | GET | Get HMT subsystem status |
| `/hmt/authority-levels` | GET | List authority levels |
| `/hmt/configure` | POST | Configure HMT parameters |
| `/hmt/pending` | GET | Get pending actions awaiting approval |
| `/hmt/approve/{id}` | POST | Approve a pending action |
| `/hmt/reject/{id}` | POST | Reject a pending action |
| `/hmt/history` | GET | Get action history |
| `/phase7/status` | GET | Get all Phase 7 subsystem statuses |
| `/launchers` | GET | List all active launchers |
| `/launchers/{id}` | GET | Get specific launcher status |
| `/ws` | WebSocket | Real-time 50Hz state stream (includes launcher state) |

### Launch Platform Events (via WebSocket)

The WebSocket stream includes launcher state with each tick:
```json
{
  "type": "state",
  "launchers": [{
    "id": "B1",
    "position": {"x": 0, "y": 0, "z": 0},
    "missiles_remaining": 3,
    "missiles_total": 4,
    "detection_range": 5000,
    "tracked_targets": [{"target_id": "T1", "range": 3500, "bearing": 45}],
    "engaged_targets": ["T1"]
  }]
}
```

The frontend `LaunchEventToast` component detects state changes and generates alerts:
- New target in `tracked_targets` → "TGT ACQUIRED"
- `missiles_remaining` decreases → "FOX THREE"
- `missiles_remaining` reaches 0 → "WINCHESTER"
- Target removed from `tracked_targets` → "TGT LOST"

## Coming Up: Phase 9

- **Multi-Domain Operations**: Air-ground-sea integration
  - Surface-to-air threats
  - Ground target prosecution
  - Maritime domain awareness
- **Electronic Warfare**: Sensor degradation and countermeasures
  - Radar jamming and ECCM
  - IR countermeasures (flares, DIRCM)
  - Stealth modeling (RCS reduction)
- **Mission Planning & Replanning**: High-level autonomy
  - Waypoint-based mission profiles
  - Dynamic replanning on threat detection
  - Fuel and weapon constraints
- **Training & Scenarios**: Educational features
  - Guided tutorials
  - Challenge scenarios with scoring
  - Performance analytics dashboard
- **Advanced Launch Platforms**: Extended bogey capabilities
  - Mobile launch platforms with movement
  - Salvo launch modes
  - Reload and resupply mechanics
  - Multi-launcher coordination
