"""
Decoy Deployment & Classification: Countermeasures for Iron Dome simulation.

Real-world adversaries deploy DECOYS to overwhelm missile defense systems.
A decoy is a lightweight object released from a real rocket that mimics
its radar signature, forcing the defense to either:

  (a) Waste interceptors on decoys (each Tamir costs ~$50,000), or
  (b) Spend precious seconds classifying targets before engaging.

This creates the core dilemma of Iron Dome threat evaluation:

  CLASSIFICATION PROBLEM:
    A rocket and its decoy look identical on radar for the first 1-2 seconds.
    But physics eventually reveals the truth:
      - Real rocket:  mass ~40kg, drag_coefficient ~0.5 → decelerates slowly
      - Decoy:        mass ~5kg,  drag_coefficient ~0.8 → decelerates FAST

  Over time (3+ seconds of tracking), the deceleration difference becomes
  statistically significant, and the system can classify with high confidence.
  This is why Iron Dome tracks threats for several seconds before committing
  interceptors — the tracking time IS the classification process.

Phase 6 integrates decoys into wave scenarios, forcing the engagement
logic to balance speed-of-response against classification accuracy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from .vector import Vec3
from .entities import Entity, EntityType, create_target
from .ballistics import FlightPhase


# ─── CONFIGURATION ────────────────────────────────────────────────

@dataclass
class DecoyConfig:
    """
    Configuration parameters for decoy deployment.

    These values define how a decoy differs from the real threat it mimics.
    The key insight: decoys are LIGHTER and DRAGGIER than real warheads,
    which is ultimately how the defense classifies them.

    Attributes:
        deploy_time:         Seconds after launch when the parent rocket
                             releases its decoys (typically during boost or
                             early ballistic phase).
        mass:                Decoy mass in kg. Much lighter than a real
                             warhead (5kg vs 40kg for a Qassam). This is
                             the primary classification discriminant.
        drag_coefficient:    Aerodynamic drag coefficient. Higher than a real
                             rocket because decoys are less streamlined
                             (often inflatable or chaff-based). Causes faster
                             deceleration, which radar tracking detects.
        radar_cross_section: Initial radar cross section in m². Starts similar
                             to the parent threat to maximize confusion, but
                             may diverge as the decoy tumbles or deflates.
    """

    deploy_time: float = 2.0
    mass: float = 5.0
    drag_coefficient: float = 0.8
    radar_cross_section: float = 0.015  # Similar to a Qassam initially


# Default configuration matching typical improvised rocket decoys
DEFAULT_DECOY_CONFIG = DecoyConfig()


# ─── CLASSIFICATION PARAMETERS ────────────────────────────────────

# Minimum tracking time (seconds) for any classification attempt
_MIN_TRACKING_TIME = 0.5

# Tracking time (seconds) at which classification reaches high confidence
_HIGH_CONFIDENCE_TIME = 3.0

# Deceleration ratio threshold: decoys decelerate >> faster than real threats
# A decoy with mass=5kg and Cd=0.8 decelerates roughly 5-8x faster than
# a real Qassam (mass=40kg, Cd=0.5) at the same speed.
_DECEL_RATIO_THRESHOLD = 2.0

# Mass threshold below which an object is likely a decoy (kg)
_MASS_DECOY_THRESHOLD = 10.0


# ─── DECOY DEPLOYMENT ─────────────────────────────────────────────

def deploy_decoys(
    parent: Entity,
    num_decoys: int,
    sim_time: float,
    config: DecoyConfig | None = None,
) -> List[Entity]:
    """
    Deploy decoy entities that branch off from a parent threat.

    When a rocket releases decoys, each decoy:
      - Starts at the parent's current position
      - Inherits the parent's velocity with small random perturbations
        (simulating ejection mechanics and tumbling)
      - Has much lower mass and higher drag than the real threat
      - Is marked as unguided with threat_type="decoy"

    The velocity perturbations are small enough that decoys initially
    track alongside the real threat on radar, but large enough that
    they gradually separate over several seconds.

    Args:
        parent:      The real threat entity releasing the decoys.
        num_decoys:  Number of decoy objects to deploy.
        sim_time:    Current simulation time (used for entity IDs).
        config:      Decoy configuration. Uses defaults if not provided.

    Returns:
        List of Entity objects representing the deployed decoys.
        Each decoy has:
          - id: "{parent.id}_D{idx}" (e.g. "W1_T0_D0", "W1_T0_D1")
          - entity_type: TARGET (appears as a target on radar)
          - threat_type: "decoy"
          - guided: False
          - engagement_decision: "" (not yet evaluated)
          - Reduced mass and increased drag per DecoyConfig
    """
    if config is None:
        config = DEFAULT_DECOY_CONFIG

    decoys: List[Entity] = []

    parent_speed = parent.velocity.magnitude()

    for idx in range(num_decoys):
        # Generate small velocity perturbation for this decoy.
        # Perturbation magnitude is ~5-10% of parent speed, applied in
        # random directions to simulate ejection scatter.
        perturbation_scale = parent_speed * 0.07  # 7% of parent speed

        # Use deterministic-ish offsets based on index to keep some spread
        # pattern, plus small random jitter for realism.
        angle = (2.0 * math.pi * idx / max(num_decoys, 1)) + random.uniform(
            -0.3, 0.3
        )
        # Perturbation is mostly lateral (perpendicular to flight path)
        # with a small component along the flight axis.
        dx = perturbation_scale * math.cos(angle)
        dy = perturbation_scale * math.sin(angle)
        dz = random.uniform(-perturbation_scale * 0.3, perturbation_scale * 0.3)

        decoy_velocity = Vec3(
            parent.velocity.x + dx,
            parent.velocity.y + dy,
            parent.velocity.z + dz,
        )

        # Small position offset so decoys don't stack exactly on the parent
        pos_offset = Vec3(
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(-1.0, 1.0),
        )
        decoy_position = parent.position + pos_offset

        decoy_id = f"{parent.id}_D{idx}"

        decoy = Entity(
            id=decoy_id,
            entity_type=EntityType.TARGET,
            position=decoy_position,
            velocity=decoy_velocity,
            acceleration=Vec3.zero(),
            max_accel=0.0,  # Decoys cannot maneuver
            mass=config.mass,
            cross_section=config.radar_cross_section,
            drag_coefficient=config.drag_coefficient,
            flight_phase=FlightPhase.BALLISTIC,  # Decoys are always ballistic
            propulsion=None,  # No motor
            burn_elapsed=0.0,
            dry_mass=config.mass,  # All mass is structure, no fuel
            guided=False,
            threat_type="decoy",
            enable_gravity=True,
            engagement_decision="",  # Not yet evaluated by threat assessment
        )

        decoys.append(decoy)

    return decoys


# ─── THREAT CLASSIFICATION ────────────────────────────────────────

def classify_threat(
    entity: Entity,
    time_tracked: float,
) -> Tuple[bool, float]:
    """
    Classify whether a tracked entity is a decoy or a real threat.

    This implements the core discrimination logic that Iron Dome uses
    to avoid wasting interceptors on decoys. The classification relies
    on two physical observables that diverge over time:

      1. MASS: Decoys are much lighter (~5kg vs ~40kg). While mass isn't
         directly observable by radar, it manifests as higher deceleration
         for a given drag profile.

      2. DECELERATION RATE: The key observable. For the same speed and
         altitude, drag force is proportional to Cd * A (drag coefficient
         times cross-section). But deceleration = F_drag / mass. So a
         lighter object with higher Cd decelerates MUCH faster.

         Real Qassam:  a_drag = F / 40kg  (with Cd=0.5)
         Decoy:        a_drag = F / 5kg   (with Cd=0.8)
         Ratio: the decoy decelerates ~6.4x faster!

    Classification confidence grows with tracking time because:
      - At t=0s: radar tracks are identical (same position, similar RCS)
      - At t=1s: slight velocity divergence, low confidence
      - At t=2s: measurable deceleration difference, moderate confidence
      - At t=3s+: statistically significant separation, high confidence

    Args:
        entity:        The entity being classified.
        time_tracked:  How long this entity has been continuously tracked
                       by the defense radar (seconds).

    Returns:
        Tuple of (is_decoy, confidence):
          - is_decoy:   True if the entity is classified as a decoy.
          - confidence: Float in [0.0, 1.0] representing classification
                        certainty. 0.0 = no information, 1.0 = certain.
    """
    # If the entity is already labeled as a decoy (e.g., in a cooperative
    # scenario or after prior classification), return immediately.
    if entity.threat_type == "decoy":
        # Even known decoys take some tracking time to confirm
        conf = min(1.0, time_tracked / _HIGH_CONFIDENCE_TIME)
        return True, conf

    # Not enough tracking time — cannot classify yet
    if time_tracked < _MIN_TRACKING_TIME:
        return False, 0.0

    # ── Physical discrimination indicators ──

    # Indicator 1: Mass-based suspicion
    # Lighter objects are more likely to be decoys.
    mass_score = 0.0
    if entity.mass < _MASS_DECOY_THRESHOLD:
        # Strongly suspicious — real threats are heavier
        mass_score = 1.0 - (entity.mass / _MASS_DECOY_THRESHOLD)
    # mass_score: 0.0 (heavy, probably real) to 1.0 (very light, likely decoy)

    # Indicator 2: Deceleration rate analysis
    # Compare the entity's drag-to-mass ratio against typical real threats.
    # A high ratio means the object is decelerating anomalously fast.
    #
    # Expected deceleration factor ~ Cd * cross_section / mass
    # Real Qassam: 0.5 * 0.015 / 40 = 0.0001875
    # Decoy:       0.8 * 0.015 / 5  = 0.0024
    # Ratio: decoy is ~12.8x higher
    decel_factor = 0.0
    if entity.mass > 0:
        decel_factor = (
            entity.drag_coefficient * entity.cross_section / entity.mass
        )

    # Reference deceleration factor for a "typical" real threat (Qassam-like)
    reference_decel_factor = 0.5 * 0.015 / 40.0  # ~0.0001875

    decel_ratio = decel_factor / reference_decel_factor if reference_decel_factor > 0 else 0.0
    decel_score = 0.0
    if decel_ratio > _DECEL_RATIO_THRESHOLD:
        # Deceleration is anomalously high — likely a decoy
        decel_score = min(1.0, (decel_ratio - _DECEL_RATIO_THRESHOLD) / 10.0)
    # decel_score: 0.0 (normal decel) to 1.0 (extreme decel, definitely decoy)

    # ── Combine indicators ──

    # Weighted combination: deceleration is the stronger signal because
    # it's directly observable via radar Doppler tracking.
    combined_score = 0.3 * mass_score + 0.7 * decel_score

    # ── Time-dependent confidence ──

    # Classification confidence increases with tracking time.
    # Early observations are noisy; longer tracking yields reliable data.
    # We model this as a sigmoid-like ramp from 0 to 1 over _HIGH_CONFIDENCE_TIME.
    time_factor = _time_confidence(time_tracked)

    # Final confidence is the product of how "decoy-like" the entity looks
    # and how long we've been watching (to confirm the observation).
    confidence = combined_score * time_factor

    # Classification decision: is_decoy if confidence exceeds threshold
    # Use a moderate threshold — erring on the side of engaging is safer
    # than letting a real threat through.
    is_decoy = confidence > 0.5

    return is_decoy, confidence


def _time_confidence(time_tracked: float) -> float:
    """
    Compute time-dependent confidence factor.

    Models the increasing certainty of classification as tracking duration
    grows. Uses a smooth ramp that reaches ~0.9 at _HIGH_CONFIDENCE_TIME
    and asymptotically approaches 1.0.

    The shape mimics real radar tracking: initial measurements are noisy,
    but Kalman filtering progressively improves state estimates.

    Args:
        time_tracked: Seconds of continuous radar tracking.

    Returns:
        Confidence factor in [0.0, 1.0].
    """
    if time_tracked <= 0:
        return 0.0

    # Saturating exponential: 1 - exp(-t / tau)
    # tau chosen so that at t=_HIGH_CONFIDENCE_TIME, factor ~= 0.95
    tau = _HIGH_CONFIDENCE_TIME / 3.0  # ~1.0 second time constant
    factor = 1.0 - math.exp(-time_tracked / tau)

    return min(1.0, factor)
