"""
Track Fusion - Multi-Sensor Data Association and Fusion

Phase 6: This module handles combining tracks from multiple sensors into
a unified picture.

KEY CONCEPTS:

1. TRACK-TO-TRACK ASSOCIATION: Determine which tracks from different
   sensors refer to the same target
   - Distance-based gating: only consider tracks within threshold
   - Statistical gating: use Mahalanobis distance considering uncertainty
   - Assignment: solve optimal association using Hungarian algorithm

2. TRACK FUSION: Combine associated tracks into a single fused track
   - Covariance Intersection (CI): conservative fusion that handles
     unknown correlations between track estimates
   - Simple weighted average: fast but assumes independence

3. TRACK MANAGEMENT:
   - Create new fused tracks when local tracks don't associate
   - Maintain track identity across sensor updates
   - Drop fused tracks when contributing sensors lose track

FUSION METHODS:

1. Simple Average: x_fused = (x1 + x2) / 2
   - Assumes equal quality, independent estimates
   - Fast but suboptimal

2. Inverse Covariance Weighted:
   x_fused = P_fused * (P1^-1 * x1 + P2^-1 * x2)
   P_fused = (P1^-1 + P2^-1)^-1
   - Optimal for independent estimates
   - Handles different uncertainties properly

3. Covariance Intersection:
   P_fused = (w * P1^-1 + (1-w) * P2^-1)^-1
   x_fused = P_fused * (w * P1^-1 * x1 + (1-w) * P2^-1 * x2)
   - w chosen to minimize trace(P_fused)
   - Conservative - valid even with unknown correlations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar

from .vector import Vec3
from .kalman import KalmanState, KalmanFilter, KalmanConfig


@dataclass
class FusionConfig:
    """Configuration for track fusion."""
    # Association parameters
    association_gate: float = 500.0  # meters - max distance to associate tracks
    mahalanobis_gate: float = 9.21   # chi-squared 3 DOF, 99% confidence

    # Fusion method: 'ci' (covariance intersection) or 'weighted' (inverse covariance)
    fusion_method: str = 'ci'

    # Track management
    min_sensors_for_fusion: int = 1   # minimum sensors to create fused track
    track_timeout: float = 5.0        # seconds before dropping stale tracks


@dataclass
class LocalTrack:
    """A track from a single sensor."""
    track_id: str
    sensor_id: str
    target_id: str
    kalman_state: KalmanState
    last_update: float
    confidence: float = 1.0


@dataclass
class FusedTrack:
    """A fused track combining multiple sensor tracks."""
    track_id: str
    target_id: str
    contributing_sensors: List[str]
    contributing_track_ids: List[str]
    fused_state: KalmanState
    confidence: float
    last_update: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "track_id": self.track_id,
            "target_id": self.target_id,
            "contributing_sensors": self.contributing_sensors,
            "contributing_track_ids": self.contributing_track_ids,
            "position": self.fused_state.get_position().to_dict(),
            "velocity": self.fused_state.get_velocity().to_dict(),
            "position_uncertainty": self.fused_state.get_position_uncertainty(),
            "confidence": self.confidence,
            "last_update": self.last_update,
            "num_updates": self.fused_state.num_updates,
        }


class TrackFusionManager:
    """
    Manages track fusion from multiple sensors.

    Workflow:
    1. Sensors report local tracks via add_local_track()
    2. Associate tracks that likely refer to same target
    3. Fuse associated tracks into unified estimates
    4. Return fused track list
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

        # Local tracks by sensor_id -> {track_id -> LocalTrack}
        self._local_tracks: Dict[str, Dict[str, LocalTrack]] = {}

        # Fused tracks by fused_track_id
        self._fused_tracks: Dict[str, FusedTrack] = {}

        # Association mapping: local_track_id -> fused_track_id
        self._associations: Dict[str, str] = {}

        # Counter for generating track IDs
        self._track_counter = 0

    def add_local_track(
        self,
        sensor_id: str,
        track_id: str,
        target_id: str,
        kalman_state: KalmanState,
        timestamp: float,
        confidence: float = 1.0
    ) -> None:
        """
        Add or update a local track from a sensor.

        Args:
            sensor_id: ID of the reporting sensor
            track_id: Local track ID from the sensor
            target_id: Target ID being tracked
            kalman_state: Kalman filter state for the track
            timestamp: Current time
            confidence: Track confidence (0-1)
        """
        if sensor_id not in self._local_tracks:
            self._local_tracks[sensor_id] = {}

        local_track = LocalTrack(
            track_id=track_id,
            sensor_id=sensor_id,
            target_id=target_id,
            kalman_state=kalman_state,
            last_update=timestamp,
            confidence=confidence
        )

        self._local_tracks[sensor_id][track_id] = local_track

    def remove_local_track(self, sensor_id: str, track_id: str) -> None:
        """Remove a local track (e.g., when sensor loses track)."""
        if sensor_id in self._local_tracks:
            self._local_tracks[sensor_id].pop(track_id, None)

    def _compute_distance(self, track1: LocalTrack, track2: LocalTrack) -> float:
        """Compute Euclidean distance between track positions."""
        p1 = track1.kalman_state.get_position()
        p2 = track2.kalman_state.get_position()
        return p1.distance_to(p2)

    def _compute_mahalanobis(self, track1: LocalTrack, track2: LocalTrack) -> float:
        """
        Compute Mahalanobis distance between tracks.

        Uses combined covariance for distance metric.
        """
        x1 = track1.kalman_state.x[:3]  # Position only
        x2 = track2.kalman_state.x[:3]
        P1 = track1.kalman_state.P[:3, :3]
        P2 = track2.kalman_state.P[:3, :3]

        # Combined covariance
        P_combined = P1 + P2
        diff = x1 - x2

        try:
            P_inv = np.linalg.inv(P_combined)
            d2 = diff @ P_inv @ diff
            return float(np.sqrt(max(0, d2)))
        except np.linalg.LinAlgError:
            # Singular matrix - fall back to Euclidean
            return float(np.linalg.norm(diff))

    def associate_tracks(self, current_time: float) -> List[List[str]]:
        """
        Associate local tracks that likely refer to the same target.

        Uses target_id for ground truth association (in simulation).
        In real systems, would use statistical gating.

        Returns:
            List of groups, where each group is [sensor1_track_id, sensor2_track_id, ...]
        """
        # Collect all active local tracks
        all_tracks: List[LocalTrack] = []
        for sensor_tracks in self._local_tracks.values():
            for track in sensor_tracks.values():
                # Only include recent tracks
                if current_time - track.last_update < self.config.track_timeout:
                    all_tracks.append(track)

        if not all_tracks:
            return []

        # Group by target_id (ground truth in simulation)
        # In real systems, this would use gating and assignment
        target_groups: Dict[str, List[LocalTrack]] = {}
        for track in all_tracks:
            if track.target_id not in target_groups:
                target_groups[track.target_id] = []
            target_groups[track.target_id].append(track)

        # Convert to list of track ID lists
        return [
            [t.track_id for t in tracks]
            for tracks in target_groups.values()
            if len(tracks) >= self.config.min_sensors_for_fusion
        ]

    def _fuse_covariance_intersection(
        self,
        tracks: List[LocalTrack]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse tracks using Covariance Intersection.

        Finds optimal weight w to minimize trace(P_fused).

        Args:
            tracks: List of tracks to fuse

        Returns:
            Tuple of (fused_state, fused_covariance)
        """
        if len(tracks) == 1:
            return tracks[0].kalman_state.x.copy(), tracks[0].kalman_state.P.copy()

        if len(tracks) == 2:
            # Two-track case - optimize weight
            x1, P1 = tracks[0].kalman_state.x, tracks[0].kalman_state.P
            x2, P2 = tracks[1].kalman_state.x, tracks[1].kalman_state.P

            def trace_objective(w):
                w = max(0.01, min(0.99, w))  # Clamp for stability
                try:
                    P1_inv = np.linalg.inv(P1)
                    P2_inv = np.linalg.inv(P2)
                    P_fused_inv = w * P1_inv + (1 - w) * P2_inv
                    P_fused = np.linalg.inv(P_fused_inv)
                    return np.trace(P_fused)
                except np.linalg.LinAlgError:
                    return 1e10

            result = minimize_scalar(trace_objective, bounds=(0.01, 0.99), method='bounded')
            w_opt = result.x

            P1_inv = np.linalg.inv(P1)
            P2_inv = np.linalg.inv(P2)
            P_fused_inv = w_opt * P1_inv + (1 - w_opt) * P2_inv
            P_fused = np.linalg.inv(P_fused_inv)
            x_fused = P_fused @ (w_opt * P1_inv @ x1 + (1 - w_opt) * P2_inv @ x2)

            return x_fused, P_fused

        # Multi-track case - equal weights
        n = len(tracks)
        w = 1.0 / n

        x_sum = np.zeros(6)
        P_inv_sum = np.zeros((6, 6))

        for track in tracks:
            try:
                P_inv = np.linalg.inv(track.kalman_state.P)
                P_inv_sum += w * P_inv
                x_sum += w * P_inv @ track.kalman_state.x
            except np.linalg.LinAlgError:
                continue

        try:
            P_fused = np.linalg.inv(P_inv_sum)
            x_fused = P_fused @ x_sum
            return x_fused, P_fused
        except np.linalg.LinAlgError:
            # Fallback to simple average
            return self._fuse_weighted_average(tracks)

    def _fuse_weighted_average(
        self,
        tracks: List[LocalTrack]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse tracks using inverse covariance weighted average.

        Optimal for independent estimates.
        """
        if len(tracks) == 1:
            return tracks[0].kalman_state.x.copy(), tracks[0].kalman_state.P.copy()

        # Inverse covariance weighting
        P_inv_sum = np.zeros((6, 6))
        x_weighted_sum = np.zeros(6)

        for track in tracks:
            try:
                P_inv = np.linalg.inv(track.kalman_state.P)
                P_inv_sum += P_inv
                x_weighted_sum += P_inv @ track.kalman_state.x
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue

        try:
            P_fused = np.linalg.inv(P_inv_sum)
            x_fused = P_fused @ x_weighted_sum
            return x_fused, P_fused
        except np.linalg.LinAlgError:
            # Complete fallback to simple average
            x_avg = np.mean([t.kalman_state.x for t in tracks], axis=0)
            P_avg = np.mean([t.kalman_state.P for t in tracks], axis=0)
            return x_avg, P_avg

    def fuse_associated_tracks(self, current_time: float) -> List[FusedTrack]:
        """
        Fuse all associated track groups.

        Args:
            current_time: Current simulation time

        Returns:
            List of fused tracks
        """
        # Get track associations
        associations = self.associate_tracks(current_time)

        fused_tracks = []

        for track_ids in associations:
            # Collect LocalTrack objects
            tracks = []
            for sensor_tracks in self._local_tracks.values():
                for track in sensor_tracks.values():
                    if track.track_id in track_ids:
                        tracks.append(track)

            if not tracks:
                continue

            # Determine target_id (should all be same)
            target_id = tracks[0].target_id

            # Fuse based on configured method
            if self.config.fusion_method == 'ci':
                x_fused, P_fused = self._fuse_covariance_intersection(tracks)
            else:
                x_fused, P_fused = self._fuse_weighted_average(tracks)

            # Create fused state
            fused_state = KalmanState(
                x=x_fused,
                P=P_fused,
                timestamp=current_time,
                num_updates=sum(t.kalman_state.num_updates for t in tracks)
            )

            # Compute average confidence
            confidence = np.mean([t.confidence for t in tracks])

            # Create or update fused track
            fused_track_id = f"FT_{target_id}"
            fused_track = FusedTrack(
                track_id=fused_track_id,
                target_id=target_id,
                contributing_sensors=[t.sensor_id for t in tracks],
                contributing_track_ids=[t.track_id for t in tracks],
                fused_state=fused_state,
                confidence=float(confidence),
                last_update=current_time
            )

            fused_tracks.append(fused_track)
            self._fused_tracks[fused_track_id] = fused_track

        return fused_tracks

    def get_fused_tracks(self) -> List[FusedTrack]:
        """Get all current fused tracks."""
        return list(self._fused_tracks.values())

    def get_fused_track(self, target_id: str) -> Optional[FusedTrack]:
        """Get fused track for a specific target."""
        fused_track_id = f"FT_{target_id}"
        return self._fused_tracks.get(fused_track_id)

    def clear(self) -> None:
        """Clear all tracks and associations."""
        self._local_tracks.clear()
        self._fused_tracks.clear()
        self._associations.clear()


def create_fusion_manager(config: Optional[FusionConfig] = None) -> TrackFusionManager:
    """Create a new track fusion manager."""
    return TrackFusionManager(config)
