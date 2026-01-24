"""
Sensor Modeling - Detection and Tracking

Phase 5: This module simulates sensor capabilities and limitations:

- Detection range: Maximum distance at which targets can be detected
- Field of View (FOV): Angular coverage of the sensor
- Detection probability: Based on range, aspect angle, and conditions
- Measurement noise: Position uncertainty from sensor accuracy
- Track quality: Confidence level based on range and measurement history

KEY CONCEPTS:

1. DETECTION PROBABILITY: Not all targets in sensor range are detected
   - Decreases with range (inverse square law for radar)
   - Affected by target aspect (RCS varies with angle)
   - Random process each update

2. FIELD OF VIEW: Sensors have limited angular coverage
   - Forward-looking sensors: typically 60-120 degree cone
   - Target must be within FOV to be detected

3. MEASUREMENT NOISE: Sensors don't give perfect information
   - Range error: typically 1-5% of range
   - Angle error: typically 0.5-2 degrees
   - Larger errors at longer range

4. TRACK QUALITY: Confidence in track based on:
   - Number of detections over time
   - Range to target
   - Measurement consistency
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math
import random

from .vector import Vec3
from .entities import Entity
from .kalman import KalmanFilter, KalmanState, KalmanConfig


@dataclass
class SensorConfig:
    """Configuration for sensor capabilities."""
    # Detection envelope
    max_range: float = 10000.0      # meters - maximum detection range
    min_range: float = 100.0        # meters - minimum detection range (too close)
    field_of_view: float = 120.0    # degrees - total cone angle

    # Detection probability
    detection_probability: float = 0.95  # at optimal range/aspect
    pd_range_falloff: float = 0.7   # probability at max_range (exponential decay)

    # Measurement noise (1-sigma)
    range_noise_std: float = 50.0   # meters - range measurement noise
    angle_noise_std: float = 1.0    # degrees - bearing/elevation noise

    # Update rate
    update_rate: float = 10.0       # Hz - how often sensor updates

    # Track management
    track_init_detections: int = 3  # detections needed to establish track
    track_coast_time: float = 2.0   # seconds before track is dropped

    # Phase 6: Kalman filter settings
    use_kalman: bool = True         # Use Kalman filter (vs alpha-beta)
    kalman_config: Optional[KalmanConfig] = None  # Kalman filter configuration


@dataclass
class Detection:
    """A single detection of a target."""
    target_id: str
    detected: bool                    # Was target detected this cycle?
    true_position: Vec3               # Actual target position (for debug)
    estimated_position: Vec3          # Position with measurement noise
    true_range: float                 # Actual range
    measured_range: float             # Range with noise
    bearing: float                    # Angle in XY plane (degrees)
    elevation: float                  # Angle from horizontal (degrees)
    confidence: float                 # 0-1 detection confidence
    in_fov: bool                      # Is target within field of view?
    timestamp: float                  # Simulation time of detection


@dataclass
class Track:
    """A tracked target with history."""
    target_id: str
    track_quality: float              # 0-1 overall track quality
    detections: int                   # Number of successful detections
    last_detection_time: float        # Time of last detection
    estimated_position: Vec3          # Filtered position estimate
    estimated_velocity: Vec3          # Filtered velocity estimate
    coasting: bool                    # True if track is coasting (no recent detection)
    is_firm: bool                     # True if track is confirmed (enough detections)

    # Phase 6: Kalman filter state (optional)
    kalman_state: Optional[KalmanState] = None

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty from Kalman state if available."""
        if self.kalman_state:
            return self.kalman_state.get_position_uncertainty()
        return 100.0  # Default uncertainty when not using Kalman


@dataclass
class SensorState:
    """State of a sensor system."""
    sensor_id: str
    owner_id: str                     # Entity that owns this sensor (interceptor ID)
    detections: List[Detection]       # Current cycle detections
    tracks: Dict[str, Track]          # Active tracks by target_id
    last_update_time: float           # Last sensor update time


class SensorModel:
    """
    Simulates a radar/seeker sensor system.

    Responsibilities:
    1. Determine if targets are within sensor envelope
    2. Calculate detection probability
    3. Generate noisy measurements
    4. Manage track quality
    5. Phase 6: Kalman filtering for optimal state estimation
    """

    def __init__(self, config: SensorConfig = None):
        self.config = config or SensorConfig()

        # Phase 6: Kalman filter instance
        if self.config.use_kalman:
            kalman_cfg = self.config.kalman_config or KalmanConfig(
                measurement_noise_pos=self.config.range_noise_std
            )
            self._kalman = KalmanFilter(kalman_cfg)
        else:
            self._kalman = None

    def is_in_fov(
        self,
        sensor_pos: Vec3,
        sensor_heading: Vec3,
        target_pos: Vec3
    ) -> bool:
        """
        Check if target is within sensor field of view.

        Args:
            sensor_pos: Sensor position
            sensor_heading: Direction sensor is pointing (usually velocity vector)
            target_pos: Target position

        Returns:
            True if target is within FOV cone
        """
        # Vector from sensor to target
        to_target = target_pos - sensor_pos
        range_to_target = to_target.magnitude()

        if range_to_target < 1.0:
            return True  # Very close, always in view

        # Check range limits
        if range_to_target < self.config.min_range:
            return False  # Too close
        if range_to_target > self.config.max_range:
            return False  # Too far

        # Check angle
        heading_mag = sensor_heading.magnitude()
        if heading_mag < 1.0:
            # No heading (stationary), assume omnidirectional
            return True

        to_target_unit = to_target.normalized()
        heading_unit = sensor_heading.normalized()

        cos_angle = to_target_unit.dot(heading_unit)
        # Clamp for numerical stability
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        # FOV is total cone, so half-angle check
        return angle_deg <= self.config.field_of_view / 2

    def compute_detection_probability(
        self,
        range_to_target: float,
        aspect_angle: float = 0.0
    ) -> float:
        """
        Compute probability of detecting target.

        Uses exponential decay with range, modified by aspect angle.

        Args:
            range_to_target: Distance to target (meters)
            aspect_angle: Target aspect angle (0=head-on, 180=tail)

        Returns:
            Probability of detection [0, 1]
        """
        if range_to_target < self.config.min_range:
            return 0.0  # Too close
        if range_to_target > self.config.max_range:
            return 0.0  # Too far

        # Range decay (exponential)
        # At range=0: Pd = detection_probability
        # At range=max_range: Pd = pd_range_falloff * detection_probability
        range_ratio = range_to_target / self.config.max_range
        # Use exponential decay
        range_factor = math.exp(-2.0 * range_ratio) * (1 - self.config.pd_range_falloff) + self.config.pd_range_falloff

        # Aspect angle effect (simplified RCS model)
        # Head-on (0 deg) and tail-on (180 deg) have higher RCS
        # Beam aspect (90 deg) has lower RCS
        aspect_rad = math.radians(aspect_angle)
        # Simplified: cos^2 gives higher probability at head-on/tail-on
        aspect_factor = 0.5 + 0.5 * abs(math.cos(aspect_rad))

        base_pd = self.config.detection_probability
        return base_pd * range_factor * aspect_factor

    def add_measurement_noise(
        self,
        true_position: Vec3,
        sensor_pos: Vec3,
        range_to_target: float
    ) -> tuple[Vec3, float]:
        """
        Add realistic measurement noise to position.

        Noise increases with range.

        Args:
            true_position: Actual target position
            sensor_pos: Sensor position
            range_to_target: True range to target

        Returns:
            Tuple of (noisy_position, noisy_range)
        """
        # Range-dependent noise scaling
        range_factor = 1.0 + (range_to_target / self.config.max_range)

        # Range noise
        range_noise = random.gauss(0, self.config.range_noise_std * range_factor)
        noisy_range = max(0, range_to_target + range_noise)

        # Angle noise in sensor frame
        bearing_noise = math.radians(random.gauss(0, self.config.angle_noise_std * range_factor))
        elevation_noise = math.radians(random.gauss(0, self.config.angle_noise_std * range_factor))

        # Convert to position offset
        # This is approximate - proper implementation would use spherical coords
        to_target = true_position - sensor_pos
        to_target_unit = to_target.normalized() if to_target.magnitude() > 0 else Vec3(1, 0, 0)

        # Create perpendicular vectors for angle noise
        # Simple approach: perturb the direction vector
        perp1 = Vec3(-to_target_unit.y, to_target_unit.x, 0)
        if perp1.magnitude() < 0.1:
            perp1 = Vec3(0, -to_target_unit.z, to_target_unit.y)
        perp1 = perp1.normalized()
        perp2 = to_target_unit.cross(perp1).normalized()

        # Apply angle noise as lateral offset
        lateral_offset = (
            perp1 * (math.tan(bearing_noise) * noisy_range) +
            perp2 * (math.tan(elevation_noise) * noisy_range)
        )

        # Noisy position
        noisy_direction = (to_target_unit * noisy_range + lateral_offset).normalized()
        noisy_position = sensor_pos + noisy_direction * noisy_range

        return noisy_position, noisy_range

    def compute_detection(
        self,
        sensor_pos: Vec3,
        sensor_vel: Vec3,
        target: Entity,
        sim_time: float,
        aspect_angle: float = 0.0
    ) -> Detection:
        """
        Compute detection result for a single target.

        Args:
            sensor_pos: Sensor position
            sensor_vel: Sensor velocity (used as heading)
            target: Target entity
            sim_time: Current simulation time
            aspect_angle: Target aspect angle (optional)

        Returns:
            Detection result
        """
        target_pos = target.position
        to_target = target_pos - sensor_pos
        true_range = to_target.magnitude()

        # Check FOV
        in_fov = self.is_in_fov(sensor_pos, sensor_vel, target_pos)

        # Compute bearing and elevation
        if true_range > 0:
            bearing = math.degrees(math.atan2(to_target.y, to_target.x))
            horizontal_dist = math.sqrt(to_target.x**2 + to_target.y**2)
            elevation = math.degrees(math.atan2(to_target.z, horizontal_dist)) if horizontal_dist > 0 else 0
        else:
            bearing = 0
            elevation = 0

        # Default values for no detection
        detected = False
        confidence = 0.0
        estimated_position = target_pos
        measured_range = true_range

        if in_fov:
            # Compute detection probability
            pd = self.compute_detection_probability(true_range, aspect_angle)

            # Roll the dice
            detected = random.random() < pd

            if detected:
                # Add measurement noise
                estimated_position, measured_range = self.add_measurement_noise(
                    target_pos, sensor_pos, true_range
                )
                # Confidence based on range and probability
                confidence = pd * (1 - true_range / self.config.max_range)
            else:
                confidence = 0.0

        return Detection(
            target_id=target.id,
            detected=detected,
            true_position=target_pos,
            estimated_position=estimated_position,
            true_range=true_range,
            measured_range=measured_range,
            bearing=bearing,
            elevation=elevation,
            confidence=confidence,
            in_fov=in_fov,
            timestamp=sim_time
        )

    def update_track(
        self,
        track: Optional[Track],
        detection: Detection,
        target_velocity: Vec3,
        sim_time: float
    ) -> Track:
        """
        Update or create a track based on new detection.

        Args:
            track: Existing track (None if new)
            detection: New detection result
            target_velocity: True target velocity (for now, would be estimated)
            sim_time: Current simulation time

        Returns:
            Updated track
        """
        # Phase 6: Use Kalman filter if enabled
        if self._kalman is not None:
            return self._update_track_kalman(track, detection, target_velocity, sim_time)

        # Original alpha-beta implementation
        return self._update_track_alpha_beta(track, detection, target_velocity, sim_time)

    def _update_track_kalman(
        self,
        track: Optional[Track],
        detection: Detection,
        target_velocity: Vec3,
        sim_time: float
    ) -> Track:
        """
        Update track using Kalman filter.

        Phase 6: Proper state estimation with uncertainty quantification.
        """
        if track is None:
            # Create new track with initial Kalman state
            if detection.detected:
                kalman_state = self._kalman.initialize(
                    detection.estimated_position,
                    target_velocity,
                    sim_time
                )
                return Track(
                    target_id=detection.target_id,
                    track_quality=detection.confidence,
                    detections=1,
                    last_detection_time=sim_time,
                    estimated_position=detection.estimated_position,
                    estimated_velocity=target_velocity,
                    coasting=False,
                    is_firm=False,
                    kalman_state=kalman_state
                )
            else:
                return Track(
                    target_id=detection.target_id,
                    track_quality=0.0,
                    detections=0,
                    last_detection_time=0.0,
                    estimated_position=detection.true_position,
                    estimated_velocity=target_velocity,
                    coasting=True,
                    is_firm=False,
                    kalman_state=None
                )

        # Update existing track
        if detection.detected:
            # Good detection - Kalman update
            new_detections = track.detections + 1

            if track.kalman_state is not None:
                # Predict and update Kalman state
                new_kalman_state = self._kalman.predict_and_update(
                    track.kalman_state,
                    detection.estimated_position,
                    sim_time,
                    measurement_noise=self.config.range_noise_std
                )
            else:
                # Initialize Kalman state
                new_kalman_state = self._kalman.initialize(
                    detection.estimated_position,
                    target_velocity,
                    sim_time
                )

            # Extract position/velocity from Kalman state
            new_position = new_kalman_state.get_position()
            new_velocity = new_kalman_state.get_velocity()

            # Update quality based on detection history and Kalman confidence
            uncertainty = new_kalman_state.get_position_uncertainty()
            # Lower uncertainty = higher quality
            uncertainty_factor = max(0, 1 - uncertainty / 500)
            quality_gain = 0.2 * (1 + uncertainty_factor)
            new_quality = min(1.0, track.track_quality + quality_gain)

            return Track(
                target_id=track.target_id,
                track_quality=new_quality,
                detections=new_detections,
                last_detection_time=sim_time,
                estimated_position=new_position,
                estimated_velocity=new_velocity,
                coasting=False,
                is_firm=new_detections >= self.config.track_init_detections,
                kalman_state=new_kalman_state
            )
        else:
            # No detection - coast track with Kalman prediction
            coast_time = sim_time - track.last_detection_time

            if track.kalman_state is not None:
                # Predict forward without update (coasting)
                dt = sim_time - track.kalman_state.timestamp
                if dt > 0:
                    predicted_kalman = self._kalman.predict(track.kalman_state, dt)
                else:
                    predicted_kalman = track.kalman_state
                predicted_position = predicted_kalman.get_position()
                predicted_velocity = predicted_kalman.get_velocity()
            else:
                # No Kalman state - dead reckoning
                dt = sim_time - track.last_detection_time if track.last_detection_time > 0 else 0
                predicted_position = track.estimated_position + track.estimated_velocity * dt
                predicted_velocity = track.estimated_velocity
                predicted_kalman = None

            # Quality degradation while coasting
            if coast_time > self.config.track_coast_time:
                new_quality = max(0.0, track.track_quality - 0.3)
            else:
                new_quality = max(0.0, track.track_quality - 0.05)

            return Track(
                target_id=track.target_id,
                track_quality=new_quality,
                detections=track.detections,
                last_detection_time=track.last_detection_time,
                estimated_position=predicted_position,
                estimated_velocity=predicted_velocity,
                coasting=True,
                is_firm=track.is_firm and new_quality > 0.3,
                kalman_state=predicted_kalman
            )

    def _update_track_alpha_beta(
        self,
        track: Optional[Track],
        detection: Detection,
        target_velocity: Vec3,
        sim_time: float
    ) -> Track:
        """
        Original alpha-beta filter implementation (backward compatible).
        """
        if track is None:
            # Create new track
            return Track(
                target_id=detection.target_id,
                track_quality=detection.confidence if detection.detected else 0.0,
                detections=1 if detection.detected else 0,
                last_detection_time=sim_time if detection.detected else 0.0,
                estimated_position=detection.estimated_position,
                estimated_velocity=target_velocity,  # Would be estimated in real system
                coasting=not detection.detected,
                is_firm=False
            )

        # Update existing track
        if detection.detected:
            # Good detection - update track
            new_detections = track.detections + 1
            # Simple alpha-beta filter for position (simplified)
            alpha = 0.3  # Position gain
            new_position = track.estimated_position + (
                (detection.estimated_position - track.estimated_position) * alpha
            )

            # Update quality based on detection history
            quality_gain = 0.2
            new_quality = min(1.0, track.track_quality + quality_gain)

            return Track(
                target_id=track.target_id,
                track_quality=new_quality,
                detections=new_detections,
                last_detection_time=sim_time,
                estimated_position=new_position,
                estimated_velocity=target_velocity,
                coasting=False,
                is_firm=new_detections >= self.config.track_init_detections
            )
        else:
            # No detection - coast track
            coast_time = sim_time - track.last_detection_time
            if coast_time > self.config.track_coast_time:
                # Track is too old, quality degraded
                new_quality = max(0.0, track.track_quality - 0.3)
            else:
                # Slight quality degradation while coasting
                new_quality = max(0.0, track.track_quality - 0.05)

            # Predict position forward (dead reckoning)
            dt = sim_time - track.last_detection_time if track.last_detection_time > 0 else 0
            predicted_position = track.estimated_position + track.estimated_velocity * dt

            return Track(
                target_id=track.target_id,
                track_quality=new_quality,
                detections=track.detections,
                last_detection_time=track.last_detection_time,
                estimated_position=predicted_position,
                estimated_velocity=track.estimated_velocity,
                coasting=True,
                is_firm=track.is_firm and new_quality > 0.3
            )


def create_sensor_state(owner_id: str, sensor_id: str = None) -> SensorState:
    """Create initial sensor state for an entity."""
    return SensorState(
        sensor_id=sensor_id or f"sensor_{owner_id}",
        owner_id=owner_id,
        detections=[],
        tracks={},
        last_update_time=0.0
    )
