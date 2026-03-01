"""
TEWA — Threat Evaluation and Weapon Assignment

Multi-layer air defense decision framework. Implements the real-world
TEWA cycle used by integrated air defense systems:

  DETECT → CLASSIFY → EVALUATE → ASSIGN → AUTHORIZE → ENGAGE

Defense layers:
  Iron Dome   — 4-70km, short-range rockets/mortars (Tamir)
  David's Sling — 40-300km, medium-range missiles/aircraft (Stunner)  
  Arrow       — 100-2400km, ballistic missiles (Arrow 3)

Key concept: Layered engagement with cascading fallback.
If the preferred tier can't engage (out of ammo, out of range, saturated),
the system falls back to the next capable tier.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple
import math

from .vector import Vec3
from .entities import Entity
from .ipp import ImpactPrediction
from .battery import Battery


class DefenseTier(str, Enum):
    IRON_DOME = "iron_dome"
    DAVIDS_SLING = "davids_sling"  
    ARROW = "arrow"


class ThreatClass(str, Enum):
    """Classified threat type for tier assignment."""
    SHORT_RANGE_ROCKET = "short_range_rocket"    # Qassam, Grad → Iron Dome
    MORTAR = "mortar"                            # Mortar → Iron Dome
    CRUISE_MISSILE = "cruise_missile"            # Cruise missile → David's Sling
    UAV = "uav"                                  # UAV → Iron Dome or David's Sling
    BALLISTIC_MISSILE = "ballistic_missile"      # Ballistic → Arrow
    UNKNOWN = "unknown"


# Tier assignment: which tier(s) can handle each threat class
# Order = preference (first is primary, rest are fallbacks)
TIER_ASSIGNMENT: Dict[ThreatClass, List[DefenseTier]] = {
    ThreatClass.SHORT_RANGE_ROCKET: [DefenseTier.IRON_DOME],
    ThreatClass.MORTAR: [DefenseTier.IRON_DOME],
    ThreatClass.CRUISE_MISSILE: [DefenseTier.DAVIDS_SLING, DefenseTier.IRON_DOME],
    ThreatClass.UAV: [DefenseTier.IRON_DOME, DefenseTier.DAVIDS_SLING],
    ThreatClass.BALLISTIC_MISSILE: [DefenseTier.ARROW, DefenseTier.DAVIDS_SLING],
    ThreatClass.UNKNOWN: [DefenseTier.IRON_DOME, DefenseTier.DAVIDS_SLING, DefenseTier.ARROW],
}

# Speed thresholds for classification (m/s)
BALLISTIC_SPEED_THRESHOLD = 2000.0    # > Mach ~6
CRUISE_SPEED_THRESHOLD = 350.0       # > ~Mach 1
SLOW_THRESHOLD = 100.0               # < 100 m/s likely UAV


@dataclass
class TEWAAssignment:
    """Result of TEWA assignment for a single threat."""
    threat_id: str
    threat_class: ThreatClass
    assigned_tier: DefenseTier
    assigned_battery_id: Optional[str]
    priority: int            # 1 = highest
    engagement_authorized: bool
    fallback_used: bool      # True if primary tier couldn't handle it
    reason: str              # Human-readable decision explanation


@dataclass
class TEWAState:
    """Current state of the TEWA system for serialization."""
    assignments: List[TEWAAssignment]
    tier_status: Dict[str, Dict]  # tier -> {batteries, total_ammo, active_engagements}
    total_threats: int
    threats_assigned: int
    fallback_count: int


def classify_threat_type(entity: Entity) -> ThreatClass:
    """
    Classify a threat based on its characteristics.
    
    Uses speed, altitude, and threat_type hint to determine classification.
    """
    # If the entity has a threat_type from the catalog, use that
    threat_type = getattr(entity, 'threat_type', None)
    if threat_type:
        mapping = {
            'qassam': ThreatClass.SHORT_RANGE_ROCKET,
            'grad': ThreatClass.SHORT_RANGE_ROCKET,
            'mortar': ThreatClass.MORTAR,
            'cruise_missile': ThreatClass.CRUISE_MISSILE,
            'uav': ThreatClass.UAV,
        }
        if threat_type in mapping:
            return mapping[threat_type]
    
    # Fallback: classify by kinematic signatures
    speed = entity.velocity.magnitude()
    altitude = entity.position.z
    
    if speed > BALLISTIC_SPEED_THRESHOLD:
        return ThreatClass.BALLISTIC_MISSILE
    elif speed > CRUISE_SPEED_THRESHOLD and altitude > 500:
        return ThreatClass.CRUISE_MISSILE
    elif speed < SLOW_THRESHOLD:
        return ThreatClass.UAV
    elif altitude > 50000:  # Very high altitude
        return ThreatClass.BALLISTIC_MISSILE
    else:
        return ThreatClass.SHORT_RANGE_ROCKET


def compute_threat_priority(
    entity: Entity,
    prediction: Optional[ImpactPrediction],
) -> int:
    """
    Compute threat priority (lower = higher priority).
    
    Factors:
    - Time to impact (closer = higher priority)
    - Whether it threatens a protected area
    - Speed (faster = higher priority)
    - Altitude (lower = more urgent, less reaction time)
    """
    priority = 100  # Default moderate priority
    
    if prediction:
        # Threatening a protected area is critical
        if prediction.engage:
            priority -= 50
        
        # Time urgency
        tti = prediction.time_to_impact
        if tti < 5.0:
            priority -= 30  # Critical — seconds away
        elif tti < 15.0:
            priority -= 20
        elif tti < 30.0:
            priority -= 10
    
    # Speed factor
    speed = entity.velocity.magnitude()
    if speed > 1000:
        priority -= 10
    elif speed > 500:
        priority -= 5
    
    return max(1, priority)


class TEWAController:
    """
    Multi-layer TEWA controller.
    
    Manages threat evaluation and weapon assignment across
    multiple defense tiers. Each tier contains one or more batteries.
    """
    
    def __init__(self):
        self.batteries: Dict[str, Battery] = {}
        self.tier_batteries: Dict[DefenseTier, List[str]] = {
            DefenseTier.IRON_DOME: [],
            DefenseTier.DAVIDS_SLING: [],
            DefenseTier.ARROW: [],
        }
        self.assignments: Dict[str, TEWAAssignment] = {}  # threat_id -> assignment
        self.fallback_count: int = 0
    
    def register_battery(self, battery: Battery) -> None:
        """Register a battery with the TEWA controller."""
        self.batteries[battery.id] = battery
        tier_str = battery.config.tier
        try:
            tier = DefenseTier(tier_str)
        except ValueError:
            tier = DefenseTier.IRON_DOME  # Default fallback
        
        if battery.id not in self.tier_batteries[tier]:
            self.tier_batteries[tier].append(battery.id)
    
    def _select_battery_in_tier(
        self,
        tier: DefenseTier,
        threat_pos: Vec3,
    ) -> Optional[str]:
        """
        Select the best battery within a tier to engage a threat.
        
        Criteria:
        1. Has ammo remaining
        2. Not at max simultaneous engagements
        3. Threat is within engagement envelope
        4. Closest battery gets priority
        """
        candidates: List[Tuple[str, float]] = []
        
        for bat_id in self.tier_batteries[tier]:
            battery = self.batteries.get(bat_id)
            if battery is None:
                continue
            
            # Check ammo
            if battery.missiles_remaining <= 0:
                continue
            
            # Check engagement capacity
            if len(battery.active_interceptors) >= battery.config.max_simultaneous:
                continue
            
            # Check range envelope
            dx = threat_pos.x - battery.config.position.x
            dy = threat_pos.y - battery.config.position.y
            dz = threat_pos.z - battery.config.position.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < battery.config.min_range or dist > battery.config.max_range:
                continue
            
            # Check radar coverage
            if not battery._is_in_radar_sector(threat_pos):
                continue
            
            candidates.append((bat_id, dist))
        
        if not candidates:
            return None
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def evaluate_and_assign(
        self,
        threats: List[Entity],
        impact_predictions: Dict[str, ImpactPrediction],
    ) -> List[TEWAAssignment]:
        """
        Run full TEWA cycle for all active threats.
        
        Returns list of assignments (one per threat that should be engaged).
        """
        results: List[TEWAAssignment] = []
        
        # Step 1: Classify and prioritize all threats
        threat_data: List[Tuple[Entity, ThreatClass, int]] = []
        for threat in threats:
            threat_class = classify_threat_type(threat)
            prediction = impact_predictions.get(threat.id)
            priority = compute_threat_priority(threat, prediction)
            threat_data.append((threat, threat_class, priority))
        
        # Sort by priority (lower number = higher priority)
        threat_data.sort(key=lambda x: x[2])
        
        # Step 2: Assign each threat to a battery
        for threat, threat_class, priority in threat_data:
            prediction = impact_predictions.get(threat.id)
            
            # Skip threats that don't need engagement (IPP says ignore)
            if prediction and not prediction.engage:
                self.assignments[threat.id] = TEWAAssignment(
                    threat_id=threat.id,
                    threat_class=threat_class,
                    assigned_tier=DefenseTier.IRON_DOME,
                    assigned_battery_id=None,
                    priority=priority,
                    engagement_authorized=False,
                    fallback_used=False,
                    reason="IPP: No threat to defended area",
                )
                continue
            
            # Already assigned and battery still has it?
            if threat.id in self.assignments:
                existing = self.assignments[threat.id]
                if existing.assigned_battery_id and existing.engagement_authorized:
                    bat = self.batteries.get(existing.assigned_battery_id)
                    if bat and bat.missiles_remaining > 0:
                        results.append(existing)
                        continue
            
            # Get tier preferences for this threat class
            tier_prefs = TIER_ASSIGNMENT.get(threat_class, [DefenseTier.IRON_DOME])
            
            assigned = False
            fallback_used = False
            
            for i, tier in enumerate(tier_prefs):
                battery_id = self._select_battery_in_tier(tier, threat.position)
                if battery_id:
                    assignment = TEWAAssignment(
                        threat_id=threat.id,
                        threat_class=threat_class,
                        assigned_tier=tier,
                        assigned_battery_id=battery_id,
                        priority=priority,
                        engagement_authorized=True,
                        fallback_used=i > 0,
                        reason=f"Assigned to {tier.value} battery {battery_id}"
                            + (" (fallback)" if i > 0 else ""),
                    )
                    self.assignments[threat.id] = assignment
                    results.append(assignment)
                    if i > 0:
                        self.fallback_count += 1
                    assigned = True
                    break
            
            if not assigned:
                # No battery available in any tier
                assignment = TEWAAssignment(
                    threat_id=threat.id,
                    threat_class=threat_class,
                    assigned_tier=tier_prefs[0] if tier_prefs else DefenseTier.IRON_DOME,
                    assigned_battery_id=None,
                    priority=priority,
                    engagement_authorized=False,
                    fallback_used=False,
                    reason="No battery available — all saturated or out of ammo",
                )
                self.assignments[threat.id] = assignment
                results.append(assignment)
        
        return results
    
    def get_state(self) -> TEWAState:
        """Serialize current TEWA state."""
        tier_status: Dict[str, Dict] = {}
        for tier in DefenseTier:
            bat_ids = self.tier_batteries[tier]
            batteries = [self.batteries[bid] for bid in bat_ids if bid in self.batteries]
            tier_status[tier.value] = {
                "batteries": len(batteries),
                "total_ammo": sum(b.missiles_remaining for b in batteries),
                "max_ammo": sum(b.missiles_total for b in batteries),
                "active_engagements": sum(len(b.active_interceptors) for b in batteries),
            }
        
        assigned_count = sum(
            1 for a in self.assignments.values() 
            if a.engagement_authorized and a.assigned_battery_id
        )
        
        return TEWAState(
            assignments=list(self.assignments.values()),
            tier_status=tier_status,
            total_threats=len(self.assignments),
            threats_assigned=assigned_count,
            fallback_count=self.fallback_count,
        )
    
    def to_event_dict(self) -> dict:
        """Serialize TEWA state for WebSocket transmission."""
        state = self.get_state()
        return {
            "assignments": [
                {
                    "threat_id": a.threat_id,
                    "threat_class": a.threat_class.value,
                    "assigned_tier": a.assigned_tier.value,
                    "assigned_battery_id": a.assigned_battery_id,
                    "priority": a.priority,
                    "engagement_authorized": a.engagement_authorized,
                    "fallback_used": a.fallback_used,
                    "reason": a.reason,
                }
                for a in state.assignments
            ],
            "tier_status": state.tier_status,
            "total_threats": state.total_threats,
            "threats_assigned": state.threats_assigned,
            "fallback_count": state.fallback_count,
        }
