"""
Wave Spawning: Timed multi-threat salvo management for Iron Dome simulation.

In real-world rocket attacks against Israel, threats rarely arrive one at a
time. Adversaries fire in WAVES (salvos) to saturate the defense:

  Wave 1 (t=0s):   5 Qassam rockets from the south, 0.5s apart
  Wave 2 (t=10s):  3 Grad rockets from the east, 1.0s apart
  Wave 3 (t=25s):  2 mortars + 4 Qassam (mixed salvo)

The WaveManager handles this complexity:
  1. Each ThreatWave defines WHEN, WHAT, and HOW MANY threats spawn.
  2. Salvo interval staggers individual rockets within a wave (realistic
     for MLRS launchers that fire tubes sequentially).
  3. Lateral spacing spreads rockets across a front, forcing the defense
     to protect a wider area.

This is Phase 6 functionality — building toward full Iron Dome scenarios
with overlapping waves, decoys, and resource-constrained interception.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .vector import Vec3
from .entities import Entity, create_threat_from_catalog


@dataclass
class ThreatWave:
    """
    Definition of a single wave (salvo) of incoming threats.

    Attributes:
        wave_id:        Unique identifier for this wave (e.g. 1, 2, 3).
        spawn_time:     Seconds into the simulation when this wave begins firing.
        threat_type:    Key into the threat catalog ("qassam", "grad", "mortar", etc.).
        count:          Number of rockets/projectiles in this wave.
        spacing:        Lateral separation in meters between adjacent threats.
                        Threats are spread perpendicular to the velocity direction
                        so they fan out across the defended area.
        start_position: Launch origin for the center of the wave (meters, ENU).
        velocity:       Initial velocity vector for all threats in this wave (m/s, ENU).
        salvo_interval: Seconds between each individual rocket launch within the wave.
                        Real MLRS systems fire tubes sequentially, not simultaneously.
                        Default 0.5s mimics a typical BM-21 Grad launcher cadence.
    """

    wave_id: int
    spawn_time: float
    threat_type: str
    count: int
    spacing: float
    start_position: Vec3
    velocity: Vec3
    salvo_interval: float = 0.5


def _compute_lateral_direction(velocity: Vec3) -> Vec3:
    """
    Compute a unit vector perpendicular to the velocity in the XY (horizontal) plane.

    This is used to spread threats laterally across the flight path.
    If the velocity is purely vertical (or zero), fall back to the East axis.

    The lateral direction is the velocity direction rotated 90 degrees
    counter-clockwise in the XY plane: if v_hat = (vx, vy), then
    lateral = (-vy, vx, 0) normalized.
    """
    vx = velocity.x
    vy = velocity.y
    horizontal_mag = math.sqrt(vx * vx + vy * vy)

    if horizontal_mag < 1e-10:
        # Velocity is purely vertical or zero — default to East axis
        return Vec3(1.0, 0.0, 0.0)

    # Rotate 90 degrees CCW in XY plane
    return Vec3(-vy / horizontal_mag, vx / horizontal_mag, 0.0)


def _compute_spawn_positions(
    center: Vec3,
    lateral: Vec3,
    count: int,
    spacing: float,
) -> List[Vec3]:
    """
    Generate spawn positions spread symmetrically around the center point.

    For count=5 and spacing=100m, positions are at offsets:
        -200, -100, 0, +100, +200  (meters along the lateral axis)

    This simulates a launcher battery spread across a front, or rockets
    aimed at slightly different points within a defended zone.
    """
    positions: List[Vec3] = []
    if count <= 0:
        return positions

    # Center the spread: offsets range from -(count-1)/2 to +(count-1)/2
    half_span = (count - 1) / 2.0
    for i in range(count):
        offset = (i - half_span) * spacing
        pos = center + lateral * offset
        positions.append(pos)

    return positions


class WaveManager:
    """
    Manages timed spawning of threat waves throughout a simulation run.

    Usage::

        waves = [
            ThreatWave(wave_id=1, spawn_time=0.0, threat_type="qassam",
                       count=5, spacing=80.0,
                       start_position=Vec3(0, -5000, 200),
                       velocity=Vec3(0, 200, 150)),
            ThreatWave(wave_id=2, spawn_time=10.0, threat_type="grad",
                       count=3, spacing=120.0,
                       start_position=Vec3(2000, -8000, 300),
                       velocity=Vec3(-50, 250, 200)),
        ]
        manager = WaveManager(waves)

        # In the simulation loop:
        new_threats = manager.update(sim_time)
        for threat in new_threats:
            simulation.add_entity(threat)

    The manager tracks which individual rockets within each wave have been
    spawned, respecting the salvo_interval so rockets do not all appear
    at once.
    """

    def __init__(self, waves: List[ThreatWave]) -> None:
        """
        Initialize the wave manager.

        Args:
            waves: List of ThreatWave definitions. Each wave will be spawned
                   once when simulation time reaches its spawn_time. Individual
                   rockets within the wave are staggered by salvo_interval.
        """
        self._waves = list(waves)

        # Track how many rockets have been spawned per wave.
        # Key: wave_id, Value: number of rockets already spawned (0..count).
        self._spawned_counts: dict[int, int] = {w.wave_id: 0 for w in waves}

        # Pre-compute lateral directions and spawn positions for each wave.
        self._lateral_dirs: dict[int, Vec3] = {}
        self._spawn_positions: dict[int, List[Vec3]] = {}

        for wave in self._waves:
            lateral = _compute_lateral_direction(wave.velocity)
            self._lateral_dirs[wave.wave_id] = lateral
            self._spawn_positions[wave.wave_id] = _compute_spawn_positions(
                center=wave.start_position,
                lateral=lateral,
                count=wave.count,
                spacing=wave.spacing,
            )

    @property
    def waves(self) -> List[ThreatWave]:
        """Read-only access to the wave definitions."""
        return list(self._waves)

    @property
    def total_threats(self) -> int:
        """Total number of individual threats across all waves."""
        return sum(w.count for w in self._waves)

    @property
    def total_spawned(self) -> int:
        """Number of threats spawned so far."""
        return sum(self._spawned_counts.values())

    @property
    def all_spawned(self) -> bool:
        """True if every rocket in every wave has been spawned."""
        return all(
            self._spawned_counts[w.wave_id] >= w.count for w in self._waves
        )

    def update(self, sim_time: float) -> List[Entity]:
        """
        Check all waves and spawn threats whose time has arrived.

        For each wave, rockets are released one at a time according to
        salvo_interval. Rocket N in a wave becomes eligible at:

            spawn_time + N * salvo_interval

        This models the sequential tube-firing of real MLRS systems.

        Args:
            sim_time: Current simulation time in seconds.

        Returns:
            List of newly spawned Entity objects (may be empty if no rockets
            are due this timestep). Each entity has:
              - id: "W{wave_id}_T{idx}" (e.g. "W1_T0", "W1_T1", ...)
              - Threat parameters from the catalog
              - Position offset by lateral spacing
        """
        newly_spawned: List[Entity] = []

        for wave in self._waves:
            already_spawned = self._spawned_counts[wave.wave_id]

            # Skip fully-spawned waves
            if already_spawned >= wave.count:
                continue

            # Check each un-spawned rocket in order
            for idx in range(already_spawned, wave.count):
                # Time at which this specific rocket should launch
                rocket_launch_time = wave.spawn_time + idx * wave.salvo_interval

                if sim_time < rocket_launch_time:
                    # This rocket (and all subsequent ones) are not ready yet
                    break

                # Spawn this rocket
                threat_id = f"W{wave.wave_id}_T{idx}"
                spawn_pos = self._spawn_positions[wave.wave_id][idx]

                entity = create_threat_from_catalog(
                    threat_type=wave.threat_type,
                    start_pos=spawn_pos,
                    velocity=wave.velocity,
                    target_id=threat_id,
                    enable_gravity=True,
                )

                newly_spawned.append(entity)
                self._spawned_counts[wave.wave_id] = idx + 1

        return newly_spawned

    def reset(self) -> None:
        """Reset all waves to un-spawned state (for simulation restarts)."""
        for wave in self._waves:
            self._spawned_counts[wave.wave_id] = 0
