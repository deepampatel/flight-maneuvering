"""
Iron Dome Battery -- Autonomous air defense unit.

A battery consists of:
- EL/M-2084 radar (360 deg coverage, 100km range)
- 3-4 launchers with ~20 Tamir interceptors each
- Battle Management & Control (BMC) -- the 'brain'

The BMC autonomously:
1. Radar detects incoming threats
2. IPP predicts impact point
3. If threat endangers protected area -> ENGAGE
4. Select launcher with best geometry
5. Launch interceptor
6. Guide to intercept

This module models the battery as a self-contained unit that can
independently detect, track, and engage threats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
import math
import random

from .vector import Vec3
from .entities import Entity, create_interceptor_from_catalog
from .ipp import (
    ProtectedArea,
    ImpactPrediction,
    check_threat_to_areas,
    compute_pk,
)
from .sensor import SensorConfig, SensorModel


class BatteryStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"       # Some launchers empty
    WINCHESTER = "winchester"    # All ammo expended
    OFFLINE = "offline"


class EngagementStatus(str, Enum):
    TRACKING = "tracking"
    ENGAGING = "engaging"
    INTERCEPTED = "intercepted"
    MISSED = "missed"


@dataclass
class LauncherUnit:
    """A single launcher pod (typically holds 20 Tamirs)."""
    id: str
    missiles_remaining: int = 20
    missiles_total: int = 20
    reload_time: float = 0.0     # Time until ready (0 = ready)
    min_launch_interval: float = 0.5  # Seconds between launches


@dataclass
class TrackedThreat:
    """A threat being tracked by the battery radar."""
    threat_id: str
    first_detected: float         # sim_time when first detected
    position: Vec3
    velocity: Vec3
    prediction: Optional[ImpactPrediction] = None
    engagement_status: EngagementStatus = EngagementStatus.TRACKING
    assigned_interceptor: Optional[str] = None
    intercept_count: int = 0      # How many interceptors launched at this threat
    max_intercepts: int = 2       # Shoot-look-shoot limit
    first_qualified_time: Optional[float] = None  # sim_time when threat first qualified for engagement
    detection_count: int = 0      # Number of radar detections (for track confirmation)
    is_firm_track: bool = False   # True after N detections confirm the track


@dataclass
class BatteryConfig:
    """Configuration for an Iron Dome battery."""
    position: Vec3                   # Battery location (meters)
    name: str = "Battery Alpha"
    tier: str = "iron_dome"          # iron_dome, davids_sling, arrow
    interceptor_type: str = "tamir"  # Catalog type for spawned interceptors

    # Radar
    radar_range: float = 100000.0    # 100 km detection range
    radar_sector: float = 360.0      # Degrees of coverage (360 = all around)
    radar_heading: float = 0.0       # Center heading of sector (degrees)
    track_capacity: int = 200        # Max simultaneous tracks

    # Launchers
    num_launchers: int = 3           # Number of launcher pods
    missiles_per_launcher: int = 20  # Missiles per pod
    min_launch_interval: float = 0.5 # Seconds between consecutive launches
    max_simultaneous: int = 6        # Max interceptors in flight

    # Engagement envelope
    min_range: float = 4000.0        # 4 km minimum engagement range
    max_range: float = 70000.0       # 70 km maximum engagement range
    min_altitude: float = 100.0      # Minimum altitude to engage

    # Protected areas this battery defends
    protected_areas: List[ProtectedArea] = field(default_factory=list)

    # Launch parameters
    launch_speed: float = 250.0      # Initial interceptor speed (m/s)
    launch_elevation: float = 80.0   # Launch angle (degrees from horizontal)

    # Operator reaction time
    operator_reaction_time: float = 2.0  # Seconds between threat qualification and launch auth

    # Radar detection model
    radar_pd: float = 0.92           # Base detection probability per scan
    track_confirm_detections: int = 3  # Detections needed before firm track


@dataclass
class BatteryState:
    """Current state of a battery for serialization."""
    id: str
    name: str
    position: Vec3
    status: BatteryStatus
    missiles_remaining: int
    missiles_total: int
    active_engagements: int
    max_simultaneous: int
    radar_range: float
    tracked_threats: int
    launchers: List[dict]            # Per-launcher status
    engagement_log: List[dict]       # Recent engagements


class Battery:
    """An autonomous air defense battery."""

    def __init__(self, battery_id: str, config: BatteryConfig):
        self.id = battery_id
        self.config = config
        self.status = BatteryStatus.OPERATIONAL

        # Create launcher pods
        self.launchers: List[LauncherUnit] = [
            LauncherUnit(
                id=f"{battery_id}_L{i+1}",
                missiles_remaining=config.missiles_per_launcher,
                missiles_total=config.missiles_per_launcher,
            )
            for i in range(config.num_launchers)
        ]

        # Radar sensor model for probabilistic detection
        self._sensor = SensorModel(SensorConfig(
            max_range=config.radar_range,
            min_range=config.min_range * 0.5,  # Detect earlier than engage
            field_of_view=config.radar_sector,
            detection_probability=config.radar_pd,
            pd_range_falloff=0.6,
            update_rate=10.0,
            track_init_detections=config.track_confirm_detections,
            use_kalman=False,  # Keep it simple — alpha-beta is fine for battery radar
        ))

        # Tracking
        self.tracked_threats: Dict[str, TrackedThreat] = {}
        self.active_interceptors: List[str] = []  # IDs of in-flight interceptors
        self.engagement_log: List[dict] = []
        self._last_launch_time: float = -999.0
        self._interceptor_counter: int = 0

    @property
    def missiles_remaining(self) -> int:
        return sum(l.missiles_remaining for l in self.launchers)

    @property
    def missiles_total(self) -> int:
        return sum(l.missiles_total for l in self.launchers)

    def _is_in_radar_sector(self, position: Vec3) -> bool:
        """Check if a position is within radar coverage."""
        dx = position.x - self.config.position.x
        dy = position.y - self.config.position.y
        distance = math.sqrt(dx * dx + dy * dy + (position.z - self.config.position.z) ** 2)

        if distance > self.config.radar_range:
            return False

        if self.config.radar_sector >= 360.0:
            return True

        # Check sector
        bearing = math.degrees(math.atan2(dy, dx)) % 360
        half_sector = self.config.radar_sector / 2
        center = self.config.radar_heading % 360
        diff = abs(bearing - center)
        if diff > 180:
            diff = 360 - diff
        return diff <= half_sector

    def _range_to_threat(self, threat_pos: Vec3) -> float:
        """Distance from battery to threat."""
        dx = threat_pos.x - self.config.position.x
        dy = threat_pos.y - self.config.position.y
        dz = threat_pos.z - self.config.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _select_launcher(self) -> Optional[LauncherUnit]:
        """Pick best launcher (most ammo remaining, ready to fire)."""
        available = [
            l for l in self.launchers
            if l.missiles_remaining > 0 and l.reload_time <= 0
        ]
        if not available:
            return None
        return max(available, key=lambda l: l.missiles_remaining)

    def _compute_launch_velocity(self, threat_pos: Vec3) -> Vec3:
        """Compute initial velocity vector for interceptor launch."""
        # Direction toward threat in XY plane
        dx = threat_pos.x - self.config.position.x
        dy = threat_pos.y - self.config.position.y
        horizontal_dist = math.sqrt(dx * dx + dy * dy)

        if horizontal_dist < 1.0:
            horizontal_dir_x, horizontal_dir_y = 1.0, 0.0
        else:
            horizontal_dir_x = dx / horizontal_dist
            horizontal_dir_y = dy / horizontal_dist

        # Launch at configured elevation angle
        elev_rad = math.radians(self.config.launch_elevation)
        horizontal_speed = self.config.launch_speed * math.cos(elev_rad)
        vertical_speed = self.config.launch_speed * math.sin(elev_rad)

        return Vec3(
            horizontal_dir_x * horizontal_speed,
            horizontal_dir_y * horizontal_speed,
            vertical_speed,
        )

    def update(
        self,
        targets: List[Entity],
        sim_time: float,
        existing_interceptors: List[Entity],
        impact_predictions: Dict[str, ImpactPrediction],
    ) -> List[Entity]:
        """
        Battery tick -- detect, evaluate, engage.

        Returns list of newly launched interceptor entities.
        """
        new_interceptors: List[Entity] = []

        # Update launcher reload timers
        for launcher in self.launchers:
            if launcher.reload_time > 0:
                launcher.reload_time -= 0.02  # dt

        # Track active interceptors still in flight
        active_interceptor_ids = set(e.id for e in existing_interceptors)
        self.active_interceptors = [
            iid for iid in self.active_interceptors if iid in active_interceptor_ids
        ]

        # --- RADAR SCAN: Detect and track threats (probabilistic) ---
        for target in targets:
            if not self._is_in_radar_sector(target.position):
                # Lost track
                if target.id in self.tracked_threats:
                    self.tracked_threats[target.id].engagement_status = EngagementStatus.MISSED
                continue

            rng = self._range_to_threat(target.position)

            # Probabilistic detection: roll against Pd based on range
            pd = self._sensor.compute_detection_probability(rng)
            detected = random.random() < pd

            if target.id not in self.tracked_threats:
                if not detected:
                    continue  # Radar missed this scan — threat not yet in track file
                # New detection
                self.tracked_threats[target.id] = TrackedThreat(
                    threat_id=target.id,
                    first_detected=sim_time,
                    position=target.position,
                    velocity=target.velocity,
                    detection_count=1,
                )
            else:
                # Existing track
                track = self.tracked_threats[target.id]
                if detected:
                    track.position = target.position
                    track.velocity = target.velocity
                    track.detection_count += 1
                    if track.detection_count >= self.config.track_confirm_detections:
                        track.is_firm_track = True
                # If not detected this scan, track coasts with stale data

            # Update IPP from engine's predictions
            if target.id in impact_predictions:
                self.tracked_threats[target.id].prediction = impact_predictions[target.id]

        # --- ENGAGEMENT DECISION: Evaluate threats and launch ---
        for threat_id, track in self.tracked_threats.items():
            # Skip already engaged or completed threats
            if track.engagement_status in [
                EngagementStatus.INTERCEPTED,
                EngagementStatus.MISSED,
            ]:
                continue

            # Skip if already at max intercept attempts
            if track.intercept_count >= track.max_intercepts:
                continue

            # Skip if already assigned an interceptor that's still flying
            if track.assigned_interceptor and track.assigned_interceptor in [
                iid for iid in self.active_interceptors
            ]:
                continue

            # Require firm track (enough detections to confirm)
            if not track.is_firm_track:
                continue

            # Check IPP: only engage if prediction says ENGAGE
            if track.prediction and not track.prediction.engage:
                continue  # Not threatening a protected area

            # Check range envelope
            rng = self._range_to_threat(track.position)
            if rng < self.config.min_range or rng > self.config.max_range:
                continue

            # Check altitude
            if track.position.z < self.config.min_altitude:
                continue

            # Operator reaction time — delay between qualification and authorization
            if track.first_qualified_time is None:
                track.first_qualified_time = sim_time
            if sim_time - track.first_qualified_time < self.config.operator_reaction_time:
                continue

            # Check max simultaneous engagements
            if len(self.active_interceptors) >= self.config.max_simultaneous:
                continue

            # Check launch interval
            if sim_time - self._last_launch_time < self.config.min_launch_interval:
                continue

            # Select launcher
            launcher = self._select_launcher()
            if launcher is None:
                continue

            # --- LAUNCH ---
            self._interceptor_counter += 1
            interceptor_id = f"{self.id}_I{self._interceptor_counter}"
            launch_vel = self._compute_launch_velocity(track.position)

            interceptor = create_interceptor_from_catalog(
                interceptor_type=self.config.interceptor_type,
                start_pos=Vec3(
                    self.config.position.x,
                    self.config.position.y,
                    self.config.position.z + 5.0,  # Slight elevation for launch
                ),
                velocity=launch_vel,
                interceptor_id=interceptor_id,
            )

            # Update state
            launcher.missiles_remaining -= 1
            launcher.reload_time = launcher.min_launch_interval
            self._last_launch_time = sim_time
            track.engagement_status = EngagementStatus.ENGAGING
            track.assigned_interceptor = interceptor_id
            track.intercept_count += 1
            self.active_interceptors.append(interceptor_id)

            self.engagement_log.append({
                "time": sim_time,
                "type": "launch",
                "threat_id": threat_id,
                "interceptor_id": interceptor_id,
                "launcher_id": launcher.id,
                "range": rng,
            })

            new_interceptors.append(interceptor)

        # Update battery status
        total = self.missiles_total
        remaining = self.missiles_remaining
        if remaining == 0:
            self.status = BatteryStatus.WINCHESTER
        elif remaining < total * 0.25:
            self.status = BatteryStatus.DEGRADED
        else:
            self.status = BatteryStatus.OPERATIONAL

        return new_interceptors

    def to_state_dict(self) -> dict:
        """Serialize battery state for frontend."""
        return {
            "id": self.id,
            "name": self.config.name,
            "position": self.config.position.to_dict(),
            "status": self.status.value,
            "tier": self.config.tier,
            "missiles_remaining": self.missiles_remaining,
            "missiles_total": self.missiles_total,
            "active_engagements": len(self.active_interceptors),
            "max_simultaneous": self.config.max_simultaneous,
            "radar_range": self.config.radar_range,
            "radar_sector": self.config.radar_sector,
            "tracked_threats": len(self.tracked_threats),
            "launchers": [
                {
                    "id": l.id,
                    "missiles_remaining": l.missiles_remaining,
                    "missiles_total": l.missiles_total,
                    "ready": l.reload_time <= 0,
                }
                for l in self.launchers
            ],
            "engagement_log": self.engagement_log[-10:],  # Last 10 events
        }


def create_iron_dome_battery(
    battery_id: str,
    position: Vec3,
    name: str = "Iron Dome Battery",
    protected_areas: Optional[List[ProtectedArea]] = None,
    missiles_per_launcher: int = 20,
    num_launchers: int = 3,
) -> Battery:
    """Convenience factory for a standard Iron Dome battery."""
    config = BatteryConfig(
        position=position,
        name=name,
        tier="iron_dome",
        interceptor_type="tamir",
        radar_range=100000.0,
        radar_sector=360.0,
        num_launchers=num_launchers,
        missiles_per_launcher=missiles_per_launcher,
        max_simultaneous=6,
        min_range=4000.0,
        max_range=70000.0,
        launch_speed=250.0,
        launch_elevation=80.0,
        protected_areas=protected_areas or [],
    )
    return Battery(battery_id, config)
