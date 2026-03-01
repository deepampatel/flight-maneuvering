# INTERCEPT

**Iron Dome-class multi-layer air defense simulation.**

Real-time 3D visualization of missile guidance, interception physics, and tactical C2 — from proportional navigation to swarm coordination to human-machine teaming.

![Intercept Simulation](docs/screenshot.png)

**React** + **Three.js** | **Python** + **FastAPI** | **WebSocket @ 50Hz** | **Docker**

---

## Quick Start

### One-command dev (recommended)

```bash
git clone https://github.com/yourusername/intercept.git
cd intercept
npm install          # installs concurrently
npm run setup        # installs frontend deps
npm run dev          # starts backend + frontend
```

Open **http://localhost:5173**. Select a scenario. Click **LAUNCH**.

### Docker (single port)

```bash
docker compose up --build
```

Open **http://localhost:8000**. Everything served from one container.

### Manual (two terminals)

```bash
# Terminal 1: Backend
cd backend
uv sync  # or: pip install -e .
python server.py

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

---

## What You Can Do

**Watch guidance laws work (or fail)**
- Pure pursuit curves hopelessly behind a turning target
- Proportional navigation drives LOS rate to zero
- Augmented PN compensates for target acceleration
- See exactly why N=3 vs N=5 matters

**Break things intentionally**
- Crank up sensor noise until tracks diverge
- Add wind and watch trajectories bend
- Enable random jinking and see intercept rates collapse
- Find the edges of the engagement envelope

**Analyze statistically**
- Monte Carlo batch runs (100+ iterations) for Pk estimation
- Engagement envelope heatmaps across range and bearing
- Record and replay any engagement

---

## Architecture

```
intercept/
├── backend/
│   ├── sim/
│   │   ├── engine.py          # Core tick loop (50Hz)
│   │   ├── guidance.py        # PN, APN, pursuit, ML
│   │   ├── evasion.py         # Target maneuvers
│   │   ├── sensor.py          # Detection + Kalman filter
│   │   ├── fusion.py          # Multi-sensor fusion
│   │   ├── assignment.py      # WTA: Greedy, Hungarian, Threat-Priority
│   │   ├── threat.py          # Threat scoring (0-100)
│   │   ├── launcher.py        # Autonomous launch platforms
│   │   ├── swarm.py           # Formation control + Reynolds flocking
│   │   ├── hmt.py             # Human-machine teaming
│   │   ├── environment.py     # Wind, drag, terrain
│   │   ├── monte_carlo.py     # Statistical analysis
│   │   └── ml/                # ONNX neural network inference
│   └── server.py              # FastAPI + WebSocket + static serving
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx             # Three.js 3D visualization
│       │   ├── MissionToolbar.tsx    # Scenario + guidance controls
│       │   ├── MissionStatusHUD.tsx  # Real-time mission panels
│       │   ├── AdvancedPanel.tsx     # Tabbed analysis sidebar
│       │   ├── SplashScreen.tsx      # Cinematic loading screen
│       │   ├── WelcomeModal.tsx      # First-visit onboarding
│       │   └── panels/              # Analysis, Recordings, ML, etc.
│       └── hooks/
│           ├── useSimulation.ts      # WebSocket + REST integration
│           └── useKeyboardShortcuts.ts
├── docker-compose.yml
├── Dockerfile
└── package.json                      # Unified dev commands
```

---

## Simulation Systems

| System | Description |
|--------|-------------|
| **Guidance** | Pure Pursuit, Proportional Nav, Augmented PN, ML Policy |
| **Evasion** | Constant-G, weave, barrel roll, random jink |
| **Sensors** | Range/angle noise, detection probability, 6-state Kalman |
| **Multi-Target** | Hungarian, Greedy, Threat-Priority WTA |
| **Launch Platforms** | Autonomous detection + lead prediction + magazine |
| **Environment** | Wind fields, gusts, altitude-dependent drag |
| **Swarm** | V, echelon, wedge, line abreast, diamond formations |
| **HMT** | Full Auto / Human-on-Loop / Human-in-Loop / Manual |
| **Analysis** | Monte Carlo (100+ runs), engagement envelope, record/replay |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Launch / Stop simulation |
| `Esc` | Abort simulation |
| `R` | Toggle recording |
| `1-4` | Camera modes (Free / Chase / Tactical / Cinematic) |
| `A` | Toggle advanced panel |
| `?` | Show shortcuts |

---

## NPM Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start backend + frontend concurrently |
| `npm run build` | Production build (frontend) |
| `npm run docker` | Docker compose up |
| `npm run typecheck` | TypeScript type check |
| `npm run lint` | ESLint |

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Three.js / React Three Fiber, Vite
- **Backend**: Python 3.11+, FastAPI, NumPy, SciPy
- **Infrastructure**: Docker, WebSocket (real-time 50Hz state streaming)
- **Optional**: ONNX Runtime (ML inference)

---

## Key Concepts

### Proportional Navigation
The core insight: if the line-of-sight angle to a target isn't changing, you're on a collision course. PN commands acceleration proportional to how fast that angle *is* changing, driving it to zero.

```
a = N x Vc x LOS_rate
```

N is the navigation constant (typically 3-5). Higher = more aggressive. Too high = oscillation.

### The Assignment Problem
With 1 interceptor and 1 target, guidance is the whole problem. With 4 interceptors and 6 targets, *who shoots what* matters as much as *how*. Hungarian algorithm gives optimal assignment but assumes you know costs.

### Human-Machine Teaming
Full autonomy is easy. Full manual is easy. The middle ground — AI acts, human overrides — requires trust calibration, workload management, and carefully designed interaction patterns.

---

## References

- Zarchan, *Tactical and Strategic Missile Guidance*
- Siouris, *Missile Guidance and Control Systems*
- Various AIAA papers on cooperative engagement

---

## License

MIT. Use it, learn from it, build on it.
