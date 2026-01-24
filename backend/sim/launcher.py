"""
Launch Platform (Bogey) Module

Models launch platforms that can detect targets and launch interceptors:
- Stationary or mobile platforms
- Radar/sensor for target detection
- Magazine of interceptors to launch
- Launch decision logic (auto or manual)

Usage:
    bogey = LaunchPlatform(
        id="B1",
        position=Vec3(0, 0, 0),
        config=LauncherConfig(
            detection_range=5000.0,
            num_missiles=4,
            launch_interval=1.0,
        )
    )

    # Each tick, check for launches
    new_interceptors = bogey.update(targets, current_time)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum
import math

from .vector import Vec3
from .entities import Entity, EntityType, create_interceptor


class LaunchMode(str, Enum):
    """Launch authorization modes."""
    AUTO = "auto"              # Launch automatically when target detected
    MANUAL = "manual"          # Wait for explicit launch command
    SALVO = "salvo"            # Launch all at once when triggered


@dataclass
class LauncherConfig:
    """Configuration for a launch platform."""
    # Sensor parameters
    detection_range: float = 5000.0      # meters - max detection range
    detection_angle: float = 360.0       # degrees - field of regard (360 = omnidirectional)
    detection_min_range: float = 100.0   # meters - minimum detection range (blind zone)

    # Magazine
    num_missiles: int = 4                # Total interceptors available

    # Launch parameters
    launch_interval: float = 0.5         # seconds between launches (rate of fire)
    launch_speed: float = 300.0          # m/s - initial interceptor speed (increased for better pursuit)
    launch_mode: LaunchMode = LaunchMode.AUTO

    # Auto-launch settings
    auto_launch_range: float = 4000.0    # Launch when target within this range
    max_simultaneous_engagements: int = 2  # Max targets to engage at once

    # Interceptor properties
    interceptor_max_accel: float = 150.0  # m/s² for launched interceptors (increased for better maneuverability)

    def to_dict(self) -> dict:
        return {
            "detection_range": self.detection_range,
            "detection_angle": self.detection_angle,
            "detection_min_range": self.detection_min_range,
            "num_missiles": self.num_missiles,
            "launch_interval": self.launch_interval,
            "launch_speed": self.launch_speed,
            "launch_mode": self.launch_mode.value,
            "auto_launch_range": self.auto_launch_range,
            "max_simultaneous_engagements": self.max_simultaneous_engagements,
        }


@dataclass
class TrackedTarget:
    """A target being tracked by the launcher."""
    target_id: str
    first_detected: float        # sim time when first detected
    last_seen: float             # sim time when last detected
    range: float                 # current range
    bearing: float               # bearing in degrees
    assigned_interceptor: Optional[str] = None  # ID of interceptor assigned to this target


@dataclass
class LaunchEvent:
    """Record of a launch event."""
    interceptor_id: str
    target_id: str
    launch_time: float
    launch_position: Vec3
    launch_velocity: Vec3


@dataclass
class LaunchPlatform:
    """
    A launch platform (bogey) that can detect targets and launch interceptors.

    Features:
    - Sensor model for target detection
    - Magazine management
    - Auto/manual launch modes
    - Track multiple targets
    """
    id: str
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3.zero)  # For mobile platforms
    config: LauncherConfig = field(default_factory=LauncherConfig)

    # State
    missiles_remaining: int = field(init=False)
    tracked_targets: Dict[str, TrackedTarget] = field(default_factory=dict)
    engaged_targets: Set[str] = field(default_factory=set)  # Targets with interceptors assigned
    launch_history: List[LaunchEvent] = field(default_factory=list)
    last_launch_time: float = field(default=-999.0)

    # Interceptor ID counter
    _interceptor_count: int = field(default=0)

    def __post_init__(self):
        self.missiles_remaining = self.config.num_missiles

    def update(
        self,
        targets: List[Entity],
        current_time: float,
        existing_interceptors: Optional[List[Entity]] = None
    ) -> List[Entity]:
        """
        Update sensor tracks and potentially launch interceptors.

        Returns:
            List of newly launched interceptors (empty if none launched)
        """
        # Update position if mobile
        # (velocity integration would happen in main sim loop)

        # Update sensor tracks
        self._update_tracks(targets, current_time)

        # Check for launches
        new_interceptors = []

        if self.config.launch_mode == LaunchMode.AUTO:
            new_interceptors = self._auto_launch(targets, current_time, existing_interceptors)

        return new_interceptors

    def _update_tracks(self, targets: List[Entity], current_time: float) -> None:
        """Update tracked targets based on sensor detection."""
        detected_ids = set()

        for target in targets:
            if self._can_detect(target):
                detected_ids.add(target.id)
                range_to_target = self.position.distance_to(target.position)
                bearing = self._compute_bearing(target.position)

                if target.id in self.tracked_targets:
                    # Update existing track
                    track = self.tracked_targets[target.id]
                    track.last_seen = current_time
                    track.range = range_to_target
                    track.bearing = bearing
                else:
                    # New track
                    self.tracked_targets[target.id] = TrackedTarget(
                        target_id=target.id,
                        first_detected=current_time,
                        last_seen=current_time,
                        range=range_to_target,
                        bearing=bearing,
                    )

        # Remove stale tracks (not seen for > 2 seconds)
        stale_threshold = current_time - 2.0
        stale_ids = [
            tid for tid, track in self.tracked_targets.items()
            if track.last_seen < stale_threshold and tid not in detected_ids
        ]
        for tid in stale_ids:
            del self.tracked_targets[tid]
            self.engaged_targets.discard(tid)

    def _can_detect(self, target: Entity) -> bool:
        """Check if target is within sensor coverage."""
        range_to_target = self.position.distance_to(target.position)

        # Range check
        if range_to_target > self.config.detection_range:
            return False
        if range_to_target < self.config.detection_min_range:
            return False

        # Angle check (if not omnidirectional)
        if self.config.detection_angle < 360.0:
            bearing = self._compute_bearing(target.position)
            half_angle = self.config.detection_angle / 2.0
            # Assuming platform faces +X direction
            if abs(bearing) > half_angle:
                return False

        return True

    def _compute_bearing(self, target_pos: Vec3) -> float:
        """Compute bearing to target in degrees (-180 to 180)."""
        delta = target_pos - self.position
        bearing_rad = math.atan2(delta.y, delta.x)
        return math.degrees(bearing_rad)

    def _auto_launch(
        self,
        targets: List[Entity],
        current_time: float,
        existing_interceptors: Optional[List[Entity]] = None
    ) -> List[Entity]:
        """Automatic launch logic."""
        new_interceptors = []

        # Check launch rate limit
        if current_time - self.last_launch_time < self.config.launch_interval:
            return new_interceptors

        # Check magazine
        if self.missiles_remaining <= 0:
            return new_interceptors

        # Check engagement limit
        if len(self.engaged_targets) >= self.config.max_simultaneous_engagements:
            return new_interceptors

        # Find targets to engage (prioritize by range)
        targets_in_range = []
        for tid, track in self.tracked_targets.items():
            if tid not in self.engaged_targets and track.range <= self.config.auto_launch_range:
                targets_in_range.append((track.range, tid, track))

        # Sort by range (closest first)
        targets_in_range.sort(key=lambda x: x[0])

        # Launch at closest unengaged target
        for _, tid, track in targets_in_range:
            if len(self.engaged_targets) >= self.config.max_simultaneous_engagements:
                break
            if self.missiles_remaining <= 0:
                break

            # Find actual target entity for launch direction
            target_entity = next((t for t in targets if t.id == tid), None)
            if target_entity is None:
                continue

            # Launch!
            interceptor = self._launch_interceptor(target_entity, current_time)
            if interceptor:
                new_interceptors.append(interceptor)
                self.engaged_targets.add(tid)
                track.assigned_interceptor = interceptor.id

        return new_interceptors

    def _launch_interceptor(self, target: Entity, current_time: float) -> Optional[Entity]:
        """Launch an interceptor toward a target with lead prediction."""
        if self.missiles_remaining <= 0:
            return None

        # Generate interceptor ID
        self._interceptor_count += 1
        interceptor_id = f"{self.id}_I{self._interceptor_count}"

        # Compute lead intercept point
        # Estimate time to intercept based on range and closing velocity
        to_target = target.position - self.position
        range_to_target = to_target.magnitude()

        # Simple lead calculation: predict where target will be
        # Use estimated flight time = range / launch_speed
        estimated_flight_time = range_to_target / self.config.launch_speed

        # Predict target position at intercept (simple linear prediction)
        predicted_pos = target.position + target.velocity * (estimated_flight_time * 0.5)

        # Compute launch direction toward predicted position
        direction = (predicted_pos - self.position).normalized()
        launch_velocity = direction * self.config.launch_speed

        # Create interceptor (Vec3 is immutable-ish, so we create new instances)
        start_pos = Vec3(self.position.x, self.position.y, self.position.z)
        interceptor = create_interceptor(
            start_pos=start_pos,
            initial_velocity=launch_velocity,
            interceptor_id=interceptor_id,
        )
        interceptor.max_accel = self.config.interceptor_max_accel

        # Record launch
        self.missiles_remaining -= 1
        self.last_launch_time = current_time
        self.launch_history.append(LaunchEvent(
            interceptor_id=interceptor_id,
            target_id=target.id,
            launch_time=current_time,
            launch_position=Vec3(self.position.x, self.position.y, self.position.z),
            launch_velocity=Vec3(launch_velocity.x, launch_velocity.y, launch_velocity.z),
        ))

        return interceptor

    def manual_launch(self, target: Entity, current_time: float) -> Optional[Entity]:
        """Manually trigger a launch at a specific target."""
        if self.missiles_remaining <= 0:
            return None

        interceptor = self._launch_interceptor(target, current_time)
        if interceptor and target.id not in self.engaged_targets:
            self.engaged_targets.add(target.id)
            if target.id in self.tracked_targets:
                self.tracked_targets[target.id].assigned_interceptor = interceptor.id

        return interceptor

    def salvo_launch(self, targets: List[Entity], current_time: float) -> List[Entity]:
        """Launch at multiple targets simultaneously (salvo mode)."""
        interceptors = []
        for target in targets[:self.missiles_remaining]:
            interceptor = self._launch_interceptor(target, current_time)
            if interceptor:
                interceptors.append(interceptor)
                self.engaged_targets.add(target.id)
        return interceptors

    def get_status(self) -> dict:
        """Get current launcher status."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "missiles_remaining": self.missiles_remaining,
            "missiles_total": self.config.num_missiles,
            "tracked_targets": len(self.tracked_targets),
            "engaged_targets": list(self.engaged_targets),
            "detection_range": self.config.detection_range,
            "launch_mode": self.config.launch_mode.value,
            "config": self.config.to_dict(),
        }

    def to_state_dict(self) -> dict:
        """Serialize for WebSocket transmission."""
        return {
            "id": self.id,
            "type": "launcher",
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "missiles_remaining": self.missiles_remaining,
            "missiles_total": self.config.num_missiles,
            "detection_range": self.config.detection_range,
            "tracked_targets": [
                {
                    "target_id": t.target_id,
                    "range": t.range,
                    "bearing": t.bearing,
                    "assigned_interceptor": t.assigned_interceptor,
                }
                for t in self.tracked_targets.values()
            ],
            "engaged_targets": list(self.engaged_targets),
        }


def create_launcher(
    position: Vec3,
    launcher_id: str = "B1",
    detection_range: float = 5000.0,
    num_missiles: int = 4,
    launch_mode: str = "auto",
) -> LaunchPlatform:
    """Factory function to create a launch platform."""
    mode = LaunchMode(launch_mode) if isinstance(launch_mode, str) else launch_mode
    # Set auto_launch_range to 80% of detection_range for reasonable engagement
    auto_launch_range = detection_range * 0.8
    config = LauncherConfig(
        detection_range=detection_range,
        num_missiles=num_missiles,
        launch_mode=mode,
        auto_launch_range=auto_launch_range,
    )
    return LaunchPlatform(
        id=launcher_id,
        position=position,
        config=config,
    )
