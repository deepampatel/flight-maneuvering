# Intercept

**Learning missile guidance by building it from scratch.**

A real-time simulation for understanding how missiles actually find their targets — from basic pursuit to proportional navigation to AI-guided intercepts.

---

## Why This Exists

I wanted to understand missile guidance beyond textbook equations. How does proportional navigation *actually* work? Why do missiles miss maneuvering targets? What makes sensor fusion hard?

So I built a sandbox to find out.

What started as "implement PN and see if it works" turned into a full tactical simulation with multi-target engagement, swarm coordination, human-machine teaming, and ML-based threat assessment. Each feature exists because I hit a question I couldn't answer without building it.

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

**Build intuition for hard problems**
- Why does WTA assignment matter with 4 targets and 3 interceptors?
- When does sensor fusion actually help vs. add latency?
- How much autonomy should a human operator delegate?

---

## The Stack

```
Frontend: React + Three.js (3D visualization)
Backend:  Python + FastAPI (simulation engine)
Comms:    WebSocket @ 50Hz (real-time state)
```

Everything runs locally. No cloud dependencies.

---

## Quick Start

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

Open http://localhost:5173. Select a scenario. Click LAUNCH. Watch.

---

## What's Implemented

### Core Simulation
- 3D kinematics with configurable timestep (default 50Hz)
- Entity model: position, velocity, acceleration, physical properties
- Semi-implicit Euler integration
- Intercept detection with configurable kill radius

### Guidance Laws
- **Pure Pursuit** — point at target (baseline, not great)
- **Proportional Navigation** — the industry standard
- **Augmented PN** — handles maneuvering targets
- **ML Policy** — ONNX neural network inference

### Target Behavior
- Constant velocity (for testing)
- Constant-G turns
- Weave/S-turns
- Barrel roll (3D evasion)
- Random jinking

### Sensors & Tracking
- Detection probability curves (range, aspect)
- Measurement noise (range, angle)
- 6-state Kalman filter per track
- Multi-sensor fusion
- Track quality scoring

### Multi-Target
- 1-20+ simultaneous targets
- Weapon-Target Assignment: Greedy, Threat-Priority, Hungarian
- Per-target threat scoring
- Real-time reassignment

### Launch Platforms
- Autonomous missile launchers ("bogeys")
- Detection range, magazine capacity
- Auto-launch with lead prediction
- Command center alerts (FOX THREE, WINCHESTER, etc.)

### Environment
- Wind fields with gusts
- Altitude-dependent air density
- Aerodynamic drag
- Terrain masking (LOS blocking)

### Swarm Coordination
- Formations: V, echelon, wedge, line abreast, trail, diamond
- Reynolds flocking behaviors
- Leader following
- Collision avoidance

### Human-Machine Teaming
- Authority levels: Full Auto → Human-in-Loop → Manual
- Action proposal/approval workflow
- Workload and trust metrics
- Configurable timeouts

### Analysis Tools
- Monte Carlo batch runs (100+ iterations)
- Parameter sweeps
- Engagement envelope mapping
- Recording & replay

---

## Project Structure

```
intercept/
├── backend/
│   ├── sim/
│   │   ├── engine.py        # Main simulation loop
│   │   ├── guidance.py      # PN, APN, pursuit, ML
│   │   ├── evasion.py       # Target maneuvers
│   │   ├── sensor.py        # Detection + Kalman
│   │   ├── fusion.py        # Multi-sensor fusion
│   │   ├── assignment.py    # WTA algorithms
│   │   ├── threat.py        # Threat scoring
│   │   ├── launcher.py      # Launch platforms
│   │   ├── swarm.py         # Formation control
│   │   ├── terrain.py       # LOS masking
│   │   ├── hmt.py           # Human-machine teaming
│   │   ├── environment.py   # Wind, drag
│   │   ├── monte_carlo.py   # Batch analysis
│   │   └── ml/              # ONNX inference
│   └── server.py            # FastAPI + WebSocket
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Scene.tsx    # Three.js 3D view
│       │   └── ...
│       └── hooks/
│           └── useSimulation.ts
└── README.md
```

---

## Key Concepts I Learned

### Proportional Navigation
The core insight: if the line-of-sight angle to a target isn't changing, you're on a collision course. PN commands acceleration proportional to how fast that angle *is* changing, driving it to zero.

```
a = N × Vc × LOS_rate
```

N is the navigation constant (typically 3-5). Higher = more aggressive maneuvering. Too high = oscillation. Too low = can't catch maneuvering targets.

### Why Sensors Make Everything Hard
Perfect state knowledge is easy. Add 50m range noise and 1° angle noise and suddenly your guidance law is chasing ghosts. The Kalman filter helps, but introduces lag. Sensor fusion helps more, but now you're correlating tracks across platforms. Every layer adds complexity.

### The Assignment Problem
With 1 interceptor and 1 target, guidance is the whole problem. With 4 interceptors and 6 targets, *who shoots what* matters as much as *how*. Hungarian algorithm gives optimal assignment but assumes you know costs. Greedy is fast but suboptimal. Threat-priority makes sense tactically but ignores geometry.

### Human-Machine Teaming
Full autonomy is easy to implement. Full manual is easy to implement. The middle — "human on the loop" where the AI acts but the human can override — requires carefully designed interaction patterns, trust calibration, and workload management. This is where real systems struggle.

---

## What's Next

Things I want to understand better:

- **Electronic warfare** — jamming, ECCM, how degraded sensors affect everything
- **Fuel constraints** — when range matters, optimal intercept changes
- **Multi-domain** — surface-to-air, naval integration
- **Distributed autonomy** — when platforms can't communicate continuously

---

## Running Tests

```bash
cd backend
python -c "from sim.engine import SimEngine; print('Engine OK')"
python -c "from sim.guidance import proportional_navigation; print('Guidance OK')"
```

---

## References

What I learned from:

- Zarchan, *Tactical and Strategic Missile Guidance* — the bible
- Siouris, *Missile Guidance and Control Systems* — good on Kalman
- Various AIAA papers on cooperative engagement

---

## License

MIT. Use it, learn from it, build on it.

---

<p align="center">
  <em>Built to learn. Kept building because the questions kept getting better.</em>
</p>
