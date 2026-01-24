"""
Weapon-Target Assignment (WTA) - Optimal Interceptor Allocation

Phase 5: This module solves the weapon-target assignment problem:
Given N interceptors and M targets, assign interceptors to targets
to maximize overall engagement effectiveness.

KEY CONCEPTS:

1. GREEDY ALGORITHMS: Fast but potentially suboptimal
   - Nearest target: Each interceptor takes closest unassigned target
   - Threat-based: Prioritize highest-threat targets

2. OPTIMAL ALGORITHMS: Find global optimum
   - Hungarian algorithm: O(n^3) optimal assignment
   - Minimizes total cost (distance, TTI, etc.)

3. COST FUNCTIONS: What we optimize
   - Range: Minimize total engagement distances
   - Time-to-intercept: Minimize total TTI
   - Threat-weighted: Balance threat level with engagement geometry

4. CONSTRAINTS:
   - One interceptor per target (for now)
   - All targets should be assigned if possible
   - Respect engagement envelopes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

from .vector import Vec3
from .entities import Entity
from .intercept import InterceptGeometry
from .threat import ThreatScore

# Avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cooperation import CooperativeEngagementManager


class WTAAlgorithm(str, Enum):
    """Available WTA algorithms."""
    GREEDY_NEAREST = "greedy_nearest"     # Nearest target first
    GREEDY_THREAT = "greedy_threat"       # Highest threat first
    HUNGARIAN = "hungarian"               # Optimal assignment
    ROUND_ROBIN = "round_robin"           # Simple rotation


@dataclass
class Assignment:
    """Single interceptor-target assignment."""
    interceptor_id: str
    target_id: str
    cost: float                     # Assignment cost (lower is better)
    reason: str                     # Why this assignment was made


@dataclass
class AssignmentResult:
    """Complete weapon-target assignment solution."""
    assignments: List[Assignment]
    total_cost: float
    algorithm: str
    unassigned_interceptors: List[str]
    unassigned_targets: List[str]
    timestamp: float = 0.0

    def get_target_for_interceptor(self, interceptor_id: str) -> Optional[str]:
        """Get assigned target for an interceptor."""
        for a in self.assignments:
            if a.interceptor_id == interceptor_id:
                return a.target_id
        return None

    def get_interceptor_for_target(self, target_id: str) -> Optional[str]:
        """Get assigned interceptor for a target."""
        for a in self.assignments:
            if a.target_id == target_id:
                return a.interceptor_id
        return None

    def to_dict(self) -> dict:
        """Serialize for JSON transmission."""
        return {
            "assignments": [
                {
                    "interceptor_id": a.interceptor_id,
                    "target_id": a.target_id,
                    "cost": round(a.cost, 2),
                    "reason": a.reason
                }
                for a in self.assignments
            ],
            "total_cost": round(self.total_cost, 2),
            "algorithm": self.algorithm,
            "unassigned_interceptors": self.unassigned_interceptors,
            "unassigned_targets": self.unassigned_targets,
            "timestamp": self.timestamp
        }


def compute_assignment_cost(
    interceptor: Entity,
    target: Entity,
    geometry: Optional[InterceptGeometry] = None,
    threat: Optional[ThreatScore] = None,
    cost_weights: Optional[Dict[str, float]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> float:
    """
    Compute cost of assigning an interceptor to a target.

    Lower cost = better assignment.

    Args:
        interceptor: Interceptor entity
        target: Target entity
        geometry: Pre-computed intercept geometry (optional)
        threat: Threat score for target (optional)
        cost_weights: Custom weights for cost components
        cooperative_manager: Cooperative engagement manager for zone costs (Phase 6)

    Returns:
        Assignment cost (lower is better)
    """
    weights = cost_weights or {
        "range": 0.4,
        "tti": 0.3,
        "threat": 0.3,
    }

    # Range cost (normalized to ~0-1 for typical scenarios)
    range_to_target = interceptor.position.distance_to(target.position)
    range_cost = range_to_target / 5000.0  # Normalize to ~1 at 5km

    # TTI cost (if geometry available)
    if geometry and geometry.time_to_intercept > 0:
        tti_cost = geometry.time_to_intercept / 30.0  # Normalize to ~1 at 30s
    else:
        # Estimate TTI from range and closing velocity
        closing_vel = interceptor.speed() + target.speed()  # Rough estimate
        tti_estimate = range_to_target / closing_vel if closing_vel > 0 else 60.0
        tti_cost = tti_estimate / 30.0

    # Threat cost (inverse - high threat = low cost)
    # We want to prioritize high-threat targets
    if threat:
        threat_cost = 1.0 - (threat.total_score / 100.0)
    else:
        threat_cost = 0.5  # Neutral if no threat data

    # Weighted sum
    total_cost = (
        weights.get("range", 0.4) * range_cost +
        weights.get("tti", 0.3) * tti_cost +
        weights.get("threat", 0.3) * threat_cost
    )

    # Phase 6: Apply zone cost modifier if cooperative manager is provided
    if cooperative_manager:
        zone_modifier = cooperative_manager.get_zone_cost_modifier(
            interceptor.id,
            target.position
        )
        total_cost *= zone_modifier

    return total_cost


def compute_cost_matrix(
    interceptors: List[Entity],
    targets: List[Entity],
    geometries: Optional[List[InterceptGeometry]] = None,
    threats: Optional[List[ThreatScore]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> List[List[float]]:
    """
    Build NxM cost matrix for assignment optimization.

    Rows = interceptors, Columns = targets.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities
        geometries: Pre-computed geometries (optional)
        threats: Threat scores for targets (optional)
        cooperative_manager: Cooperative manager for zone costs (Phase 6)

    Returns:
        2D cost matrix [num_interceptors][num_targets]
    """
    cost_matrix = []

    for interceptor in interceptors:
        row = []
        for target in targets:
            # Find matching geometry if available
            geometry = None
            if geometries:
                for g in geometries:
                    if g.interceptor_id == interceptor.id and g.target_id == target.id:
                        geometry = g
                        break

            # Find matching threat score if available
            threat = None
            if threats:
                for t in threats:
                    if t.target_id == target.id:
                        threat = t
                        break

            cost = compute_assignment_cost(
                interceptor, target, geometry, threat,
                cooperative_manager=cooperative_manager
            )
            row.append(cost)

        cost_matrix.append(row)

    return cost_matrix


def greedy_nearest_assignment(
    interceptors: List[Entity],
    targets: List[Entity],
    geometries: Optional[List[InterceptGeometry]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> AssignmentResult:
    """
    Greedy assignment: each interceptor takes nearest unassigned target.

    Fast O(N*M) algorithm. May not be globally optimal.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities
        geometries: Pre-computed geometries (optional)
        cooperative_manager: Cooperative manager for zone costs (Phase 6)

    Returns:
        AssignmentResult with greedy assignments
    """
    assignments = []
    assigned_targets = set()
    assigned_interceptors = set()
    total_cost = 0.0

    # Sort interceptors by some priority (could be customized)
    remaining_interceptors = list(interceptors)

    while remaining_interceptors:
        interceptor = remaining_interceptors.pop(0)

        # Find best unassigned target (considering zone costs)
        best_target = None
        best_cost = float('inf')

        for target in targets:
            if target.id in assigned_targets:
                continue

            # Compute cost including zone modifier
            cost = compute_assignment_cost(
                interceptor, target,
                cooperative_manager=cooperative_manager
            )
            if cost < best_cost:
                best_cost = cost
                best_target = target

        if best_target:
            assignments.append(Assignment(
                interceptor_id=interceptor.id,
                target_id=best_target.id,
                cost=best_cost,
                reason="nearest_available"
            ))
            total_cost += best_cost
            assigned_targets.add(best_target.id)
            assigned_interceptors.add(interceptor.id)

    # Track unassigned
    unassigned_interceptors = [i.id for i in interceptors if i.id not in assigned_interceptors]
    unassigned_targets = [t.id for t in targets if t.id not in assigned_targets]

    return AssignmentResult(
        assignments=assignments,
        total_cost=total_cost,
        algorithm=WTAAlgorithm.GREEDY_NEAREST.value,
        unassigned_interceptors=unassigned_interceptors,
        unassigned_targets=unassigned_targets
    )


def greedy_threat_assignment(
    interceptors: List[Entity],
    targets: List[Entity],
    threats: List[ThreatScore],
    geometries: Optional[List[InterceptGeometry]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> AssignmentResult:
    """
    Threat-prioritized assignment: assign to highest-threat targets first.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities
        threats: Threat scores for each target
        geometries: Pre-computed geometries (optional)
        cooperative_manager: Cooperative manager for zone costs (Phase 6)

    Returns:
        AssignmentResult with threat-prioritized assignments
    """
    assignments = []
    assigned_targets = set()
    assigned_interceptors = set()
    total_cost = 0.0

    # Sort targets by threat score (highest first)
    threat_map = {t.target_id: t for t in threats}
    sorted_targets = sorted(
        targets,
        key=lambda t: threat_map.get(t.id, ThreatScore(
            target_id=t.id, total_score=0, threat_level='low',
            time_score=0, closing_score=0, aspect_score=0,
            altitude_score=0, maneuver_score=0, time_to_impact=0,
            closing_velocity=0, aspect_angle=0, altitude_delta=0
        )).total_score,
        reverse=True
    )

    for target in sorted_targets:
        if not interceptors or all(i.id in assigned_interceptors for i in interceptors):
            break

        # Find best available interceptor for this high-threat target
        best_interceptor = None
        best_cost = float('inf')

        for interceptor in interceptors:
            if interceptor.id in assigned_interceptors:
                continue

            cost = compute_assignment_cost(
                interceptor, target,
                cooperative_manager=cooperative_manager
            )
            if cost < best_cost:
                best_cost = cost
                best_interceptor = interceptor

        if best_interceptor:
            threat_info = threat_map.get(target.id)
            reason = f"threat_priority_rank_{threat_info.priority_rank}" if threat_info else "threat_priority"

            assignments.append(Assignment(
                interceptor_id=best_interceptor.id,
                target_id=target.id,
                cost=best_cost,
                reason=reason
            ))
            total_cost += best_cost
            assigned_targets.add(target.id)
            assigned_interceptors.add(best_interceptor.id)

    # Track unassigned
    unassigned_interceptors = [i.id for i in interceptors if i.id not in assigned_interceptors]
    unassigned_targets = [t.id for t in targets if t.id not in assigned_targets]

    return AssignmentResult(
        assignments=assignments,
        total_cost=total_cost,
        algorithm=WTAAlgorithm.GREEDY_THREAT.value,
        unassigned_interceptors=unassigned_interceptors,
        unassigned_targets=unassigned_targets
    )


def hungarian_assignment(
    interceptors: List[Entity],
    targets: List[Entity],
    cost_matrix: Optional[List[List[float]]] = None,
    geometries: Optional[List[InterceptGeometry]] = None,
    threats: Optional[List[ThreatScore]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> AssignmentResult:
    """
    Optimal assignment using Hungarian algorithm.

    O(n^3) complexity but guarantees global optimum.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities
        cost_matrix: Pre-computed cost matrix (optional)
        geometries: Pre-computed geometries (optional)
        threats: Threat scores (optional)
        cooperative_manager: Cooperative manager for zone costs (Phase 6)

    Returns:
        AssignmentResult with optimal assignments
    """
    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except ImportError:
        has_scipy = False

    if not has_scipy:
        # Fallback to greedy if scipy not available
        return greedy_nearest_assignment(interceptors, targets, geometries, cooperative_manager)

    # Build cost matrix if not provided
    if cost_matrix is None:
        cost_matrix = compute_cost_matrix(
            interceptors, targets, geometries, threats,
            cooperative_manager=cooperative_manager
        )

    # Handle rectangular matrices (different number of interceptors/targets)
    n_interceptors = len(interceptors)
    n_targets = len(targets)

    if n_interceptors == 0 or n_targets == 0:
        return AssignmentResult(
            assignments=[],
            total_cost=0.0,
            algorithm=WTAAlgorithm.HUNGARIAN.value,
            unassigned_interceptors=[i.id for i in interceptors],
            unassigned_targets=[t.id for t in targets]
        )

    # Convert to numpy array for scipy
    import numpy as np
    cost_array = np.array(cost_matrix)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_array)

    # Build assignments
    assignments = []
    total_cost = 0.0
    assigned_interceptors = set()
    assigned_targets = set()

    for i, j in zip(row_ind, col_ind):
        if i < n_interceptors and j < n_targets:
            cost = cost_array[i, j]
            assignments.append(Assignment(
                interceptor_id=interceptors[i].id,
                target_id=targets[j].id,
                cost=float(cost),
                reason="optimal_hungarian"
            ))
            total_cost += cost
            assigned_interceptors.add(interceptors[i].id)
            assigned_targets.add(targets[j].id)

    unassigned_interceptors = [i.id for i in interceptors if i.id not in assigned_interceptors]
    unassigned_targets = [t.id for t in targets if t.id not in assigned_targets]

    return AssignmentResult(
        assignments=assignments,
        total_cost=float(total_cost),
        algorithm=WTAAlgorithm.HUNGARIAN.value,
        unassigned_interceptors=unassigned_interceptors,
        unassigned_targets=unassigned_targets
    )


def round_robin_assignment(
    interceptors: List[Entity],
    targets: List[Entity]
) -> AssignmentResult:
    """
    Simple round-robin assignment.

    Assigns interceptors to targets in order. Useful as baseline.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities

    Returns:
        AssignmentResult with round-robin assignments
    """
    assignments = []
    total_cost = 0.0

    n_assignments = min(len(interceptors), len(targets))

    for i in range(n_assignments):
        interceptor = interceptors[i]
        target = targets[i]
        cost = compute_assignment_cost(interceptor, target)

        assignments.append(Assignment(
            interceptor_id=interceptor.id,
            target_id=target.id,
            cost=cost,
            reason="round_robin"
        ))
        total_cost += cost

    unassigned_interceptors = [interceptors[i].id for i in range(n_assignments, len(interceptors))]
    unassigned_targets = [targets[i].id for i in range(n_assignments, len(targets))]

    return AssignmentResult(
        assignments=assignments,
        total_cost=total_cost,
        algorithm=WTAAlgorithm.ROUND_ROBIN.value,
        unassigned_interceptors=unassigned_interceptors,
        unassigned_targets=unassigned_targets
    )


def compute_assignment(
    interceptors: List[Entity],
    targets: List[Entity],
    algorithm: WTAAlgorithm = WTAAlgorithm.GREEDY_NEAREST,
    geometries: Optional[List[InterceptGeometry]] = None,
    threats: Optional[List[ThreatScore]] = None,
    cooperative_manager: Optional['CooperativeEngagementManager'] = None
) -> AssignmentResult:
    """
    Main entry point for weapon-target assignment.

    Args:
        interceptors: List of interceptor entities
        targets: List of target entities
        algorithm: Which WTA algorithm to use
        geometries: Pre-computed intercept geometries
        threats: Pre-computed threat scores
        cooperative_manager: Cooperative engagement manager for zone-aware assignment (Phase 6)

    Returns:
        AssignmentResult with computed assignments
    """
    import time

    if algorithm == WTAAlgorithm.GREEDY_NEAREST:
        result = greedy_nearest_assignment(interceptors, targets, geometries, cooperative_manager)
    elif algorithm == WTAAlgorithm.GREEDY_THREAT:
        if threats:
            result = greedy_threat_assignment(interceptors, targets, threats, geometries, cooperative_manager)
        else:
            # Fallback to nearest if no threat data
            result = greedy_nearest_assignment(interceptors, targets, geometries, cooperative_manager)
    elif algorithm == WTAAlgorithm.HUNGARIAN:
        result = hungarian_assignment(interceptors, targets, None, geometries, threats, cooperative_manager)
    elif algorithm == WTAAlgorithm.ROUND_ROBIN:
        result = round_robin_assignment(interceptors, targets)
    else:
        result = greedy_nearest_assignment(interceptors, targets, geometries, cooperative_manager)

    result.timestamp = time.time()
    return result
