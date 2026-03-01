# INTERCEPT — Scenario Builder & Advanced UX Plan

## Goal
Transform INTERCEPT into a **developer playground** where users can configure, present, and experiment with different air defense scenarios — with full control over defense layers, threat waves, and engagement parameters.

---

## Current Gaps

| What | Backend supports | Frontend exposes | Gap |
|------|-----------------|-----------------|-----|
| Battery config (tier, position, radar, ammo, range) | `BatteryConfig` dataclass (16 fields) | Hardcoded per scenario | **Full config UI needed** |
| Wave spawning (timing, count, threat type) | `ThreatWave` in `EngineConfig` | Hardcoded per scenario | **Wave designer needed** |
| Protected areas (position, radius, priority) | `ProtectedArea` in `EngineConfig` | Hardcoded per scenario | **Zone editor needed** |
| Threat types (qassam, grad, cruise_missile) | Entity catalog in engine | Not selectable | **Threat picker needed** |
| Custom batteries via API | NOT in `RunConfig` | N/A | **Backend extension needed** |
| Custom waves via API | NOT in `RunConfig` | N/A | **Backend extension needed** |
| Custom protected areas via API | Only via `custom_zones` (basic) | Only in MissionPlanner | **Needs upgrade** |
| HMT authority levels | 4 levels in backend | Hidden | Expose in UI |
| Terrain masking | Backend ready | Hidden toggle | Expose in UI |
| Datalink simulation | Backend ready | Hidden toggle | Expose in UI |

---

## Architecture: "Scenario Builder" Drawer

Replace the current scattered controls (toolbar dropdowns + ADV sidebar + mission planner) with a unified **Scenario Builder** — a slide-out left drawer with progressive disclosure.

### Layout
```
┌─────────────────────────────────────────────────────┐
│  INTERCEPT              [Online]   v2.0              │ ← header (keep)
├────────┬────────────────────────────────────────────┤
│        │                                             │
│ SCEN.  │                                             │
│ BUILD  │          3D SCENE (hero)                    │
│ DRAWER │                                             │
│        │                                             │
│ ~320px │                                             │
│        │                                             │
│        ├─────────────────────────────────────────────┤
│        │  Status HUD panels  │  Timeline             │
└────────┴────────────────────────────────────────────┘
```

The drawer opens from the left with a toggle button. When closed, the 3D scene gets full width. When running a sim, the drawer auto-collapses to not obscure the action.

### Drawer Sections (collapsible accordion)

**1. Quick Start** (default open)
- Preset scenario cards (the 15 narratives) as a scrollable grid
- Click a card → auto-fills all sections below
- "Custom" card to start blank

**2. Defense Layers** (the big new feature)
- Add/remove batteries via `+ Add Battery` button
- Each battery card:
  - **Tier selector**: Iron Dome / David's Sling / Arrow (radio buttons with color badges)
  - Tier presets auto-fill sensible defaults (range, ammo, speed, etc.)
  - **Position**: X/Z inputs + "Place on Map" button (click in 3D scene)
  - **Radar**: Range slider (with tier-appropriate min/max), Sector slider (90°–360°)
  - **Ammo**: Missiles per launcher × Number of launchers (with total shown)
  - **Engagement envelope**: Min/Max range (constrained by tier)
  - Collapsible "Advanced" section for: launch speed, launch elevation, min altitude, max simultaneous
- Visual: batteries appear as colored circles on the 3D ground plane in real-time as you configure

**3. Threat Waves**
- List of wave cards, each with:
  - **Timing**: Delay (seconds from sim start)
  - **Threat type**: qassam / grad / cruise_missile (with speed/alt presets shown)
  - **Count**: Number of threats in this wave (1–20)
  - **Spawn direction**: Bearing slider (0–360°) or "Random"
  - **Decoys**: Toggle + percentage slider
- `+ Add Wave` button
- Quick presets: "Single Rocket", "Salvo (5)", "Saturation (15 in 3 waves)"

**4. Protected Areas**
- List of zone cards:
  - Name, Position (X/Z), Radius, Priority (1–10)
  - Color picker
- `+ Add Zone` button
- Visual: green circles on ground plane

**5. Engagement Settings** (replaces MissionToolbar dropdowns)
- Guidance Law: dropdown
- Nav Constant: slider
- WTA Algorithm: dropdown (now always visible)
- Kill Radius: slider

**6. Environment** (replaces AdvancedPanel > Environment tab)
- Wind: speed + direction + gusts
- Drag toggle
- Terrain toggle
- Datalink toggle

**7. Advanced** (collapsed by default)
- HMT authority level
- Cooperative engagement toggle
- Swarm formation (if interceptors > 1)
- Simulation dt, max_time

### Action Bar (sticky bottom of drawer)
```
┌─────────────────────────────┐
│  [▶ Launch]    [REC]  [ADV] │
└─────────────────────────────┘
```

---

## Implementation Steps

### Phase A: Backend — Accept Custom Batteries & Waves (server.py + engine.py)

**A1. Extend `RunConfig` with battery/wave/area configs**

In `backend/server.py`, add new Pydantic models:

```python
class BatteryConfigModel(BaseModel):
    """Battery configuration from frontend."""
    id: str
    name: str = "Battery"
    tier: str = "iron_dome"  # iron_dome, davids_sling, arrow
    position: Vec3Model
    radar_range: float = 70000.0
    radar_sector: float = 360.0
    num_launchers: int = 3
    missiles_per_launcher: int = 20
    max_simultaneous: int = 6
    min_range: float = 4000.0
    max_range: float = 70000.0
    launch_speed: float = 250.0
    launch_elevation: float = 80.0
    min_altitude: float = 100.0
    protected_area_ids: list[str] = []  # Link to custom zones

class WaveConfigModel(BaseModel):
    """Wave configuration from frontend."""
    delay: float = 0.0          # Seconds from sim start
    threat_type: str = "qassam" # qassam, grad, cruise_missile
    count: int = 3
    spawn_bearing: float = 0.0  # Degrees, 0=North
    spawn_range: float = 15000.0
    spawn_altitude: float = 2000.0
    decoy_fraction: float = 0.0 # 0.0–1.0

class ProtectedAreaModel(BaseModel):
    """Protected area from frontend."""
    id: str
    name: str = "City"
    center: Vec3Model
    radius: float = 2000.0
    priority: int = 5           # 1–10
```

Add to `RunConfig`:
```python
custom_batteries: Optional[list[BatteryConfigModel]] = None
custom_waves: Optional[list[WaveConfigModel]] = None
custom_protected_areas: Optional[list[ProtectedAreaModel]] = None
```

**A2. Wire into engine setup** (`backend/server.py` POST /runs handler)

After existing custom_entities handling, add:
- Convert `BatteryConfigModel` → `BatteryConfig` dataclass and attach to engine
- Convert `WaveConfigModel` → `ThreatWave` and attach to wave manager
- Convert `ProtectedAreaModel` → `ProtectedArea`

**A3. Add GET /api/catalog endpoint**

Return available options for the frontend:
```json
{
  "tiers": {
    "iron_dome": { "name": "Iron Dome", "range": [4000, 70000], "interceptor": "tamir", "defaults": {...} },
    "davids_sling": { "name": "David's Sling", "range": [40000, 300000], "interceptor": "stunner", "defaults": {...} },
    "arrow": { "name": "Arrow", "range": [100000, 2400000], "interceptor": "arrow_3", "defaults": {...} }
  },
  "threat_types": {
    "qassam": { "name": "Qassam", "speed": 200, "altitude": 2000, "category": "short_range" },
    "grad": { "name": "Grad", "speed": 300, "altitude": 3000, "category": "medium_range" },
    "cruise_missile": { "name": "Cruise Missile", "speed": 250, "altitude": 500, "category": "guided" }
  },
  "guidance_laws": [...],
  "evasion_types": [...],
  "wta_algorithms": [...]
}
```

### Phase B: Frontend — Scenario Builder Component

**B1. Create `ScenarioBuilder.tsx`** — the main drawer component
- Accordion sections using a simple collapsible pattern
- State management for all config (batteries, waves, areas, settings)
- Exports a `buildRunConfig()` method that produces a valid RunConfig

**B2. Create section components:**
- `ScenarioPresets.tsx` — card grid of narrative scenarios + "Custom" option
- `DefenseLayerEditor.tsx` — battery list with add/remove/configure
- `BatteryCard.tsx` — individual battery configuration card
- `ThreatWaveEditor.tsx` — wave list with add/remove/configure
- `WaveCard.tsx` — individual wave card
- `ProtectedAreaEditor.tsx` — zone list
- `EngagementSettings.tsx` — guidance, nav constant, WTA, kill radius
- `EnvironmentSettings.tsx` — wind, drag, terrain, datalink

**B3. 3D Scene Integration**
- Show battery positions as colored rings on the ground plane (real-time preview)
- Show protected areas as translucent green domes
- Show wave spawn directions as directional arrows
- "Place on Map" mode: click in 3D scene to set position

**B4. Wire into App.tsx**
- Replace current MissionToolbar + AdvancedPanel with ScenarioBuilder drawer
- Keep StatusBar, MissionStatusHUD, and runtime overlays as-is
- Drawer collapses when sim is running

### Phase C: Styling

- Glassmorphic drawer panel matching existing design system
- Tier color badges: Iron Dome = blue (#3b82f6), David's Sling = cyan (#06b6d4), Arrow = violet (#8b5cf6)
- Collapsible sections with smooth animations
- Card-based layout for batteries/waves (not a dense form)
- Responsive: drawer can be resized or collapsed

---

## File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `frontend/src/components/ScenarioBuilder.tsx` | Main drawer component |
| `frontend/src/components/builder/ScenarioPresets.tsx` | Preset cards |
| `frontend/src/components/builder/DefenseLayerEditor.tsx` | Battery list |
| `frontend/src/components/builder/BatteryCard.tsx` | Single battery config |
| `frontend/src/components/builder/ThreatWaveEditor.tsx` | Wave list |
| `frontend/src/components/builder/WaveCard.tsx` | Single wave config |
| `frontend/src/components/builder/ProtectedAreaEditor.tsx` | Zone list |
| `frontend/src/components/builder/EngagementSettings.tsx` | Guidance/WTA/etc |
| `frontend/src/components/builder/EnvironmentSettings.tsx` | Wind/terrain/etc |
| `frontend/src/data/tierDefaults.ts` | Tier preset values |

### Modified Files
| File | Changes |
|------|---------|
| `backend/server.py` | Add BatteryConfigModel, WaveConfigModel, ProtectedAreaModel; extend RunConfig; add /api/catalog endpoint |
| `backend/sim/engine.py` | Accept custom batteries/waves from RunConfig |
| `frontend/src/App.tsx` | Replace toolbar/advanced with ScenarioBuilder drawer; add drawer toggle state |
| `frontend/src/App.css` | Add drawer styles, battery card styles, wave card styles |
| `frontend/src/types.ts` | Add BatteryConfig, WaveConfig, ProtectedAreaConfig, CatalogResponse types |
| `frontend/src/hooks/useSimulation.ts` | Update startRun to include new config fields |

### Kept As-Is
- `MissionStatusHUD.tsx` — runtime HUD (works during sim)
- `EngagementTimeline.tsx` — timeline bar
- `ThreatBoard.tsx` — threat table
- `LayerDiagram.tsx` — defense layer visualization
- `Scene.tsx` — 3D rendering (minor additions for battery/zone preview)
- All narrative scenarios in `scenarios.ts` (used as presets)

---

## Priority Order

1. **Phase A** (backend): ~2 hours — unblocks everything
2. **Phase B1-B2** (drawer + sections): ~4 hours — core UX
3. **Phase B3** (3D preview): ~2 hours — visual feedback
4. **Phase B4** (wiring): ~1 hour — integration
5. **Phase C** (polish): ~1 hour — styling refinements
