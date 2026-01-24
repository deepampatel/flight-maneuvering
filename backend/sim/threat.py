"""
Threat Assessment - Target Prioritization and Scoring

This module scores and ranks threats based on multiple factors:
- Time to impact: how soon will the target reach us
- Closing velocity: how fast is the gap closing
- Aspect angle: head-on threats are more dangerous
- Altitude advantage: targets above us have energy advantage
- Maneuverability: highly maneuvering targets are harder to intercept

SCORING PHILOSOPHY:

A threat score of 0-100 where:
- 80-100: CRITICAL - Immediate engagement required
- 60-79:  HIGH     - Prioritize for engagement
- 40-59:  MEDIUM   - Monitor, engage if resources allow
- 0-39:   LOW      - Track but low priority

Each component contributes a weighted score (0-1) that is
normalized into the final 0-100 score.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
import math

from .vector import Vec3
from .entities import Entity
from .intercept import InterceptGeometry, compute_intercept_geometry

if TYPE_CHECKING:
    from .ml import ThreatModel


@dataclass
class ThreatWeights:
    """
    Configurable weights for threat scoring components.

    All weights should sum to 1.0 for proper normalization.
    Default weights prioritize:
    1. Time to impact (most important - imminent threats first)
    2. Closing velocity (fast approaching threats)
    3. Aspect angle (head-on more threatening)
    4. Altitude (energy advantage)
    5. Maneuverability (evasive targets harder to deal with)
    """
    time_to_impact: float = 0.35
    closing_velocity: float = 0.25
    aspect_angle: float = 0.20
    altitude_advantage: float = 0.10
    maneuverability: float = 0.10

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (self.time_to_impact + self.closing_velocity +
                 self.aspect_angle + self.altitude_advantage +
                 self.maneuverability)
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.time_to_impact /= total
            self.closing_velocity /= total
            self.aspect_angle /= total
            self.altitude_advantage /= total
            self.maneuverability /= total


@dataclass
class ThreatScore:
    """
    Computed threat assessment for a single target.
    """
    target_id: str

    # Overall score (0-100)
    total_score: float
    threat_level: str  # 'critical', 'high', 'medium', 'low'

    # Component scores (0-1 each, before weighting)
    time_score: float
    closing_score: float
    aspect_score: float
    altitude_score: float
    maneuver_score: float

    # Raw values used in scoring
    time_to_impact: float     # seconds
    closing_velocity: float   # m/s
    aspect_angle: float       # degrees
    altitude_delta: float     # meters (positive = target above)

    # Engagement recommendation
    priority_rank: int = 1    # 1 = highest priority

    def to_dict(self) -> dict:
        """Serialize for JSON transmission."""
        return {
            "target_id": self.target_id,
            "total_score": round(self.total_score, 1),
            "threat_level": self.threat_level,
            "time_score": round(self.time_score, 3),
            "closing_score": round(self.closing_score, 3),
            "aspect_score": round(self.aspect_score, 3),
            "altitude_score": round(self.altitude_score, 3),
            "maneuver_score": round(self.maneuver_score, 3),
            "time_to_impact": round(self.time_to_impact, 2),
            "closing_velocity": round(self.closing_velocity, 1),
            "aspect_angle": round(self.aspect_angle, 1),
            "altitude_delta": round(self.altitude_delta, 1),
            "priority_rank": self.priority_rank
        }


@dataclass
class ThreatAssessment:
    """
    Complete threat picture from one interceptor's perspective.
    """
    timestamp: float
    assessor_id: str  # Which interceptor is assessing
    threats: List[ThreatScore]
    highest_threat_id: str
    engagement_recommendation: str  # 'engage', 'monitor', 'ignore'

    def to_dict(self) -> dict:
        """Serialize for JSON transmission."""
        return {
            "timestamp": self.timestamp,
            "assessor_id": self.assessor_id,
            "threats": [t.to_dict() for t in self.threats],
            "highest_threat_id": self.highest_threat_id,
            "engagement_recommendation": self.engagement_recommendation
        }


def get_threat_level(score: float) -> str:
    """Map numeric score (0-100) to threat level string."""
    if score >= 80:
        return 'critical'
    elif score >= 60:
        return 'high'
    elif score >= 40:
        return 'medium'
    else:
        return 'low'


def score_time_to_impact(tti: float) -> float:
    """
    Score based on time to impact.

    Higher score for imminent threats (shorter time).
    Uses exponential decay - threat drops off quickly after ~30s.

    Score mapping:
    - 0s:  1.0 (immediate threat)
    - 10s: ~0.72
    - 30s: ~0.37
    - 60s: ~0.14
    - 120s: ~0.02

    Args:
        tti: Time to impact in seconds (-1 if not closing)

    Returns:
        Score from 0 to 1
    """
    if tti < 0:
        return 0.0  # Not closing, not a threat

    # Exponential decay with tau = 30 seconds
    tau = 30.0
    return math.exp(-tti / tau)


def score_closing_velocity(closing_vel: float, max_expected: float = 500.0) -> float:
    """
    Score based on closing velocity.

    Higher closing velocity = more threatening.
    Linear scaling up to max_expected.

    Args:
        closing_vel: Closing velocity in m/s
        max_expected: Expected maximum closing velocity

    Returns:
        Score from 0 to 1
    """
    if closing_vel <= 0:
        return 0.0  # Opening, not closing

    # Linear scaling, clamped to [0, 1]
    return min(1.0, closing_vel / max_expected)


def score_aspect_angle(aspect: float) -> float:
    """
    Score based on aspect angle.

    Head-on (0°) is most threatening - gives least reaction time.
    Tail-chase (180°) is least threatening - we're chasing them.

    Uses cosine curve for smooth transition:
    - 0°: 1.0 (head-on)
    - 90°: 0.5 (beam)
    - 180°: 0.0 (tail)

    Args:
        aspect: Aspect angle in degrees (0-180)

    Returns:
        Score from 0 to 1
    """
    # Normalize to 0-180
    aspect = max(0.0, min(180.0, aspect))

    # Cosine gives smooth 1->0 transition over 0->180
    # (1 + cos(aspect)) / 2 maps [0,180] to [1,0]
    return (1.0 + math.cos(math.radians(aspect))) / 2.0


def score_altitude_advantage(altitude_delta: float, max_delta: float = 1000.0) -> float:
    """
    Score based on altitude difference.

    Targets above us have energy advantage (can dive for speed).
    Uses sigmoid for smooth transition around zero.

    Args:
        altitude_delta: Target altitude minus our altitude (positive = above us)
        max_delta: Delta at which score saturates

    Returns:
        Score from 0 to 1 (0.5 = same altitude)
    """
    # Sigmoid centered at 0
    # At max_delta, we want ~0.88
    # At -max_delta, we want ~0.12
    k = 3.0 / max_delta  # Steepness

    # Sigmoid: 1 / (1 + e^(-k*x))
    return 1.0 / (1.0 + math.exp(-k * altitude_delta))


def score_maneuverability(accel_magnitude: float, max_accel: float = 100.0) -> float:
    """
    Score based on target's current acceleration.

    Highly maneuvering targets are harder to intercept.
    Linear scaling up to max expected acceleration.

    Args:
        accel_magnitude: Target's current acceleration magnitude (m/s²)
        max_accel: Expected maximum acceleration

    Returns:
        Score from 0 to 1
    """
    if accel_magnitude < 0.1:
        return 0.0  # Not maneuvering

    return min(1.0, accel_magnitude / max_accel)


def compute_threat_score(
    interceptor: Entity,
    target: Entity,
    geometry: InterceptGeometry,
    weights: Optional[ThreatWeights] = None
) -> ThreatScore:
    """
    Compute threat score for a single target.

    Args:
        interceptor: Our interceptor
        target: Target to assess
        geometry: Pre-computed intercept geometry
        weights: Custom weights (default weights if None)

    Returns:
        ThreatScore with all component scores and total
    """
    weights = weights or ThreatWeights()

    # Compute component scores
    time_score = score_time_to_impact(geometry.time_to_intercept)
    closing_score = score_closing_velocity(geometry.closing_velocity)
    aspect_score = score_aspect_angle(geometry.aspect_angle)

    # Altitude difference (positive = target above)
    altitude_delta = target.position.z - interceptor.position.z
    altitude_score = score_altitude_advantage(altitude_delta)

    # Maneuverability from current acceleration
    maneuver_score = score_maneuverability(target.acceleration.magnitude())

    # Weighted sum
    weighted_total = (
        weights.time_to_impact * time_score +
        weights.closing_velocity * closing_score +
        weights.aspect_angle * aspect_score +
        weights.altitude_advantage * altitude_score +
        weights.maneuverability * maneuver_score
    )

    # Scale to 0-100
    total_score = weighted_total * 100.0

    # Determine threat level
    threat_level = get_threat_level(total_score)

    return ThreatScore(
        target_id=target.id,
        total_score=total_score,
        threat_level=threat_level,
        time_score=time_score,
        closing_score=closing_score,
        aspect_score=aspect_score,
        altitude_score=altitude_score,
        maneuver_score=maneuver_score,
        time_to_impact=geometry.time_to_intercept,
        closing_velocity=geometry.closing_velocity,
        aspect_angle=geometry.aspect_angle,
        altitude_delta=altitude_delta
    )


def assess_all_threats(
    interceptor: Entity,
    targets: List[Entity],
    geometries: List[InterceptGeometry],
    weights: Optional[ThreatWeights] = None
) -> ThreatAssessment:
    """
    Assess and rank all threats from one interceptor's perspective.

    Args:
        interceptor: Our interceptor
        targets: List of targets to assess
        geometries: Pre-computed geometry for each target
        weights: Custom weights (default if None)

    Returns:
        ThreatAssessment with ranked threats and recommendation
    """
    import time

    # Compute scores for all targets
    scores: List[ThreatScore] = []
    for target, geometry in zip(targets, geometries):
        score = compute_threat_score(interceptor, target, geometry, weights)
        scores.append(score)

    # Sort by total score (descending - highest threat first)
    scores.sort(key=lambda s: s.total_score, reverse=True)

    # Assign priority ranks
    for i, score in enumerate(scores):
        score.priority_rank = i + 1

    # Determine highest threat
    highest_threat_id = scores[0].target_id if scores else ""
    highest_score = scores[0].total_score if scores else 0.0

    # Engagement recommendation
    if highest_score >= 60:
        recommendation = 'engage'
    elif highest_score >= 40:
        recommendation = 'monitor'
    else:
        recommendation = 'ignore'

    return ThreatAssessment(
        timestamp=time.time(),
        assessor_id=interceptor.id,
        threats=scores,
        highest_threat_id=highest_threat_id,
        engagement_recommendation=recommendation
    )


def quick_threat_assessment(
    interceptor: Entity,
    target: Entity,
    weights: Optional[ThreatWeights] = None
) -> ThreatScore:
    """
    Quick single-target threat assessment.

    Computes geometry and threat score in one call.
    Use this for real-time updates.

    Args:
        interceptor: Our interceptor
        target: Target to assess
        weights: Custom weights

    Returns:
        ThreatScore for the target
    """
    geometry = compute_intercept_geometry(interceptor, target)
    return compute_threat_score(interceptor, target, geometry, weights)


# -----------------------------------------------------------------------------
# ML-Based Threat Assessment
# -----------------------------------------------------------------------------

def ml_threat_assessment(
    interceptor: Entity,
    targets: List[Entity],
    model: "ThreatModel",
    geometries: Optional[List[InterceptGeometry]] = None,
) -> ThreatAssessment:
    """
    Assess threats using ML model.

    Uses neural network to predict threat scores instead of rule-based
    weighted scoring. Falls back to rule-based if model not loaded.

    Args:
        interceptor: Our interceptor
        targets: List of targets to assess
        model: ThreatModel instance (from ml.inference)
        geometries: Pre-computed geometries (optional)

    Returns:
        ThreatAssessment with ML-predicted scores
    """
    import time
    from .ml.features import extract_threat_features, extract_batch_threat_features

    # Compute geometries if not provided
    if geometries is None:
        geometries = [compute_intercept_geometry(interceptor, t) for t in targets]

    # Extract features for all targets
    feature_batch = extract_batch_threat_features(interceptor, targets, geometries)
    target_ids = [t.id for t in targets]

    # Get ML predictions
    predictions = model.predict_batch(feature_batch, target_ids)

    # Convert to ThreatScore objects
    scores: List[ThreatScore] = []
    for i, (pred, target, geom) in enumerate(zip(predictions, targets, geometries)):
        # Fill in the full ThreatScore with geometry data
        altitude_delta = target.position.z - interceptor.position.z

        # Compute component scores (for display/debugging)
        time_score = score_time_to_impact(geom.time_to_intercept)
        closing_score = score_closing_velocity(geom.closing_velocity)
        aspect_score = score_aspect_angle(geom.aspect_angle)
        altitude_score = score_altitude_advantage(altitude_delta)
        maneuver_score = score_maneuverability(target.acceleration.magnitude())

        score = ThreatScore(
            target_id=pred.target_id,
            total_score=pred.threat_score,
            threat_level=pred.threat_level,
            time_score=time_score,
            closing_score=closing_score,
            aspect_score=aspect_score,
            altitude_score=altitude_score,
            maneuver_score=maneuver_score,
            time_to_impact=geom.time_to_intercept,
            closing_velocity=geom.closing_velocity,
            aspect_angle=geom.aspect_angle,
            altitude_delta=altitude_delta,
        )
        scores.append(score)

    # Sort by total score (descending)
    scores.sort(key=lambda s: s.total_score, reverse=True)

    # Assign priority ranks
    for i, score in enumerate(scores):
        score.priority_rank = i + 1

    # Determine highest threat
    highest_threat_id = scores[0].target_id if scores else ""
    highest_score = scores[0].total_score if scores else 0.0

    # Engagement recommendation
    if highest_score >= 60:
        recommendation = 'engage'
    elif highest_score >= 40:
        recommendation = 'monitor'
    else:
        recommendation = 'ignore'

    return ThreatAssessment(
        timestamp=time.time(),
        assessor_id=interceptor.id,
        threats=scores,
        highest_threat_id=highest_threat_id,
        engagement_recommendation=recommendation
    )


def hybrid_threat_assessment(
    interceptor: Entity,
    targets: List[Entity],
    model: Optional["ThreatModel"] = None,
    weights: Optional[ThreatWeights] = None,
    ml_weight: float = 0.5,
) -> ThreatAssessment:
    """
    Hybrid threat assessment combining rule-based and ML approaches.

    Blends the scores from both methods for robust assessment.

    Args:
        interceptor: Our interceptor
        targets: List of targets
        model: ThreatModel instance (optional)
        weights: Custom weights for rule-based
        ml_weight: Weight for ML score (0-1), rule-based gets (1 - ml_weight)

    Returns:
        ThreatAssessment with blended scores
    """
    import time

    # Compute geometries once
    geometries = [compute_intercept_geometry(interceptor, t) for t in targets]

    # Get rule-based assessment
    rule_assessment = assess_all_threats(interceptor, targets, geometries, weights)

    # If no model or model not loaded, just return rule-based
    if model is None or not model.model_loaded:
        return rule_assessment

    # Get ML assessment
    ml_assessment = ml_threat_assessment(interceptor, targets, model, geometries)

    # Create lookup for ML scores
    ml_scores = {s.target_id: s.total_score for s in ml_assessment.threats}

    # Blend scores
    blended_scores: List[ThreatScore] = []
    for rule_score in rule_assessment.threats:
        ml_score = ml_scores.get(rule_score.target_id, rule_score.total_score)
        blended = rule_score.total_score * (1 - ml_weight) + ml_score * ml_weight

        # Create new ThreatScore with blended total
        new_score = ThreatScore(
            target_id=rule_score.target_id,
            total_score=blended,
            threat_level=get_threat_level(blended),
            time_score=rule_score.time_score,
            closing_score=rule_score.closing_score,
            aspect_score=rule_score.aspect_score,
            altitude_score=rule_score.altitude_score,
            maneuver_score=rule_score.maneuver_score,
            time_to_impact=rule_score.time_to_impact,
            closing_velocity=rule_score.closing_velocity,
            aspect_angle=rule_score.aspect_angle,
            altitude_delta=rule_score.altitude_delta,
        )
        blended_scores.append(new_score)

    # Re-sort and rank
    blended_scores.sort(key=lambda s: s.total_score, reverse=True)
    for i, score in enumerate(blended_scores):
        score.priority_rank = i + 1

    highest_threat_id = blended_scores[0].target_id if blended_scores else ""
    highest_score = blended_scores[0].total_score if blended_scores else 0.0

    if highest_score >= 60:
        recommendation = 'engage'
    elif highest_score >= 40:
        recommendation = 'monitor'
    else:
        recommendation = 'ignore'

    return ThreatAssessment(
        timestamp=time.time(),
        assessor_id=interceptor.id,
        threats=blended_scores,
        highest_threat_id=highest_threat_id,
        engagement_recommendation=recommendation
    )
