"""
Cooperative Engagement Module

Provides multi-platform coordination including:
- Engagement zones (killboxes) for sector defense
- Target handoff between interceptors
- Coordinated assignment considering zones
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import uuid
import math

from .entities import Vec3


class HandoffStatus(str, Enum):
    """Status of a handoff request."""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class HandoffReason(str, Enum):
    """Reason for requesting a handoff."""
    FUEL_LOW = "fuel_low"
    OUT_OF_ENVELOPE = "out_of_envelope"
    REASSIGNMENT = "reassignment"
    ZONE_BOUNDARY = "zone_boundary"
    BETTER_GEOMETRY = "better_geometry"
    MANUAL = "manual"


@dataclass
class EngagementZone:
    """
    Defines a 3D engagement zone (killbox) for sector defense.

    Interceptors assigned to a zone prioritize targets within their zone.
    Zones can overlap - priority determines which zone "owns" a target.
    """
    zone_id: str
    name: str
    center: Vec3
    dimensions: Vec3  # width (x), depth (y), height (z)
    rotation: float = 0.0  # heading in degrees (0=North, 90=East)
    assigned_interceptors: List[str] = field(default_factory=list)
    priority: int = 1  # Higher = more important
    active: bool = True
    color: str = "#00ff00"  # For visualization

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside this zone (axis-aligned after rotation)."""
        # Translate to zone-centered coordinates
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        # Rotate point to zone's local frame
        angle_rad = math.radians(-self.rotation)
        local_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        local_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        local_z = dz

        # Check bounds
        half_w = self.dimensions.x / 2
        half_d = self.dimensions.y / 2
        half_h = self.dimensions.z / 2

        return (abs(local_x) <= half_w and
                abs(local_y) <= half_d and
                abs(local_z) <= half_h)

    def distance_to_center(self, point: Vec3) -> float:
        """Distance from point to zone center."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "center": {"x": self.center.x, "y": self.center.y, "z": self.center.z},
            "dimensions": {"x": self.dimensions.x, "y": self.dimensions.y, "z": self.dimensions.z},
            "rotation": self.rotation,
            "assigned_interceptors": self.assigned_interceptors,
            "priority": self.priority,
            "active": self.active,
            "color": self.color,
        }


@dataclass
class HandoffRequest:
    """
    Request to transfer target engagement from one interceptor to another.
    """
    request_id: str
    from_interceptor: str
    to_interceptor: str
    target_id: str
    reason: HandoffReason
    status: HandoffStatus = HandoffStatus.PENDING
    timestamp: float = 0.0
    approved_at: Optional[float] = None
    executed_at: Optional[float] = None
    expiry_time: float = 30.0  # Seconds before request expires

    def is_expired(self, current_time: float) -> bool:
        """Check if request has expired."""
        return (self.status == HandoffStatus.PENDING and
                current_time - self.timestamp > self.expiry_time)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "from_interceptor": self.from_interceptor,
            "to_interceptor": self.to_interceptor,
            "target_id": self.target_id,
            "reason": self.reason.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "approved_at": self.approved_at,
            "executed_at": self.executed_at,
            "expiry_time": self.expiry_time,
        }


@dataclass
class CooperativeState:
    """Current state of cooperative engagement system."""
    engagement_zones: Dict[str, EngagementZone] = field(default_factory=dict)
    pending_handoffs: Dict[str, HandoffRequest] = field(default_factory=dict)
    completed_handoffs: List[HandoffRequest] = field(default_factory=list)
    interceptor_zones: Dict[str, str] = field(default_factory=dict)  # interceptor_id -> zone_id
    target_assignments: Dict[str, str] = field(default_factory=dict)  # target_id -> interceptor_id

    def to_dict(self) -> dict:
        return {
            "engagement_zones": [z.to_dict() for z in self.engagement_zones.values()],
            "pending_handoffs": [h.to_dict() for h in self.pending_handoffs.values()],
            "completed_handoffs": [h.to_dict() for h in self.completed_handoffs[-10:]],  # Last 10
            "interceptor_zones": self.interceptor_zones,
            "target_assignments": self.target_assignments,
        }


class CooperativeEngagementManager:
    """
    Manages cooperative engagement between multiple interceptors.

    Responsibilities:
    - Zone management (create, delete, assign interceptors)
    - Handoff coordination (request, approve, execute)
    - Zone-aware target assignment
    """

    def __init__(self):
        self.state = CooperativeState()
        self.auto_approve_handoffs = True
        self.handoff_cooldown: Dict[str, float] = {}  # interceptor_id -> last handoff time
        self.cooldown_period = 5.0  # Seconds between handoffs for same interceptor

    def reset(self):
        """Reset cooperative state."""
        self.state = CooperativeState()
        self.handoff_cooldown.clear()

    # ================== Zone Management ==================

    def create_zone(
        self,
        name: str,
        center: Vec3,
        dimensions: Vec3,
        rotation: float = 0.0,
        priority: int = 1,
        color: str = "#00ff00"
    ) -> str:
        """Create a new engagement zone."""
        zone_id = f"zone_{uuid.uuid4().hex[:8]}"
        zone = EngagementZone(
            zone_id=zone_id,
            name=name,
            center=center,
            dimensions=dimensions,
            rotation=rotation,
            priority=priority,
            color=color,
        )
        self.state.engagement_zones[zone_id] = zone
        return zone_id

    def delete_zone(self, zone_id: str) -> bool:
        """Delete an engagement zone."""
        if zone_id not in self.state.engagement_zones:
            return False

        # Unassign interceptors from this zone
        zone = self.state.engagement_zones[zone_id]
        for interceptor_id in zone.assigned_interceptors:
            if self.state.interceptor_zones.get(interceptor_id) == zone_id:
                del self.state.interceptor_zones[interceptor_id]

        del self.state.engagement_zones[zone_id]
        return True

    def get_zone(self, zone_id: str) -> Optional[EngagementZone]:
        """Get zone by ID."""
        return self.state.engagement_zones.get(zone_id)

    def get_zones(self) -> List[EngagementZone]:
        """Get all active zones."""
        return [z for z in self.state.engagement_zones.values() if z.active]

    def assign_interceptor_to_zone(self, interceptor_id: str, zone_id: str) -> bool:
        """Assign an interceptor to a zone."""
        if zone_id not in self.state.engagement_zones:
            return False

        # Remove from previous zone
        old_zone_id = self.state.interceptor_zones.get(interceptor_id)
        if old_zone_id and old_zone_id in self.state.engagement_zones:
            old_zone = self.state.engagement_zones[old_zone_id]
            if interceptor_id in old_zone.assigned_interceptors:
                old_zone.assigned_interceptors.remove(interceptor_id)

        # Add to new zone
        zone = self.state.engagement_zones[zone_id]
        if interceptor_id not in zone.assigned_interceptors:
            zone.assigned_interceptors.append(interceptor_id)
        self.state.interceptor_zones[interceptor_id] = zone_id

        return True

    def unassign_interceptor(self, interceptor_id: str) -> bool:
        """Remove interceptor from its current zone."""
        zone_id = self.state.interceptor_zones.get(interceptor_id)
        if not zone_id:
            return False

        if zone_id in self.state.engagement_zones:
            zone = self.state.engagement_zones[zone_id]
            if interceptor_id in zone.assigned_interceptors:
                zone.assigned_interceptors.remove(interceptor_id)

        del self.state.interceptor_zones[interceptor_id]
        return True

    def find_zone_for_point(self, point: Vec3) -> Optional[EngagementZone]:
        """Find the highest-priority active zone containing the point."""
        containing_zones = [
            z for z in self.state.engagement_zones.values()
            if z.active and z.contains_point(point)
        ]

        if not containing_zones:
            return None

        # Return highest priority zone
        return max(containing_zones, key=lambda z: z.priority)

    # ================== Handoff Management ==================

    def request_handoff(
        self,
        from_interceptor: str,
        to_interceptor: str,
        target_id: str,
        reason: HandoffReason,
        current_time: float
    ) -> Optional[str]:
        """
        Request a target handoff between interceptors.
        Returns request_id if successful, None if rejected.
        """
        # Check cooldown
        last_handoff = self.handoff_cooldown.get(from_interceptor, 0)
        if current_time - last_handoff < self.cooldown_period:
            return None

        # Check if from_interceptor is actually assigned to target
        if self.state.target_assignments.get(target_id) != from_interceptor:
            return None

        # Check if there's already a pending handoff for this target
        for request in self.state.pending_handoffs.values():
            if request.target_id == target_id and request.status == HandoffStatus.PENDING:
                return None

        request_id = f"handoff_{uuid.uuid4().hex[:8]}"
        request = HandoffRequest(
            request_id=request_id,
            from_interceptor=from_interceptor,
            to_interceptor=to_interceptor,
            target_id=target_id,
            reason=reason,
            timestamp=current_time,
        )

        self.state.pending_handoffs[request_id] = request

        # Auto-approve if enabled
        if self.auto_approve_handoffs:
            self.approve_handoff(request_id, current_time)

        return request_id

    def approve_handoff(self, request_id: str, current_time: float) -> bool:
        """Approve a pending handoff request."""
        request = self.state.pending_handoffs.get(request_id)
        if not request or request.status != HandoffStatus.PENDING:
            return False

        if request.is_expired(current_time):
            request.status = HandoffStatus.EXPIRED
            return False

        request.status = HandoffStatus.APPROVED
        request.approved_at = current_time
        return True

    def reject_handoff(self, request_id: str) -> bool:
        """Reject a pending handoff request."""
        request = self.state.pending_handoffs.get(request_id)
        if not request or request.status != HandoffStatus.PENDING:
            return False

        request.status = HandoffStatus.REJECTED
        return True

    def execute_handoff(self, request_id: str, current_time: float) -> bool:
        """Execute an approved handoff."""
        request = self.state.pending_handoffs.get(request_id)
        if not request or request.status != HandoffStatus.APPROVED:
            return False

        # Update target assignment
        self.state.target_assignments[request.target_id] = request.to_interceptor

        # Update request status
        request.status = HandoffStatus.EXECUTED
        request.executed_at = current_time

        # Update cooldown
        self.handoff_cooldown[request.from_interceptor] = current_time

        # Move to completed
        self.state.completed_handoffs.append(request)
        del self.state.pending_handoffs[request_id]

        return True

    def get_pending_handoffs(self) -> List[HandoffRequest]:
        """Get all pending handoff requests."""
        return [h for h in self.state.pending_handoffs.values()
                if h.status == HandoffStatus.PENDING]

    def get_approved_handoffs(self) -> List[HandoffRequest]:
        """Get all approved but not executed handoffs."""
        return [h for h in self.state.pending_handoffs.values()
                if h.status == HandoffStatus.APPROVED]

    def cleanup_expired_requests(self, current_time: float):
        """Mark expired requests."""
        for request in self.state.pending_handoffs.values():
            if request.is_expired(current_time):
                request.status = HandoffStatus.EXPIRED

    # ================== Assignment Integration ==================

    def set_target_assignment(self, target_id: str, interceptor_id: str):
        """Set the current target-interceptor assignment."""
        self.state.target_assignments[target_id] = interceptor_id

    def get_target_assignment(self, target_id: str) -> Optional[str]:
        """Get the interceptor assigned to a target."""
        return self.state.target_assignments.get(target_id)

    def clear_target_assignment(self, target_id: str):
        """Clear assignment for a target (e.g., after intercept)."""
        if target_id in self.state.target_assignments:
            del self.state.target_assignments[target_id]

    def get_zone_cost_modifier(
        self,
        interceptor_id: str,
        target_position: Vec3
    ) -> float:
        """
        Get cost modifier for zone-aware assignment.

        Returns a multiplier (1.0 = normal, <1.0 = preferred, >1.0 = discouraged).
        Interceptors prefer targets in their assigned zone.
        """
        interceptor_zone_id = self.state.interceptor_zones.get(interceptor_id)

        if not interceptor_zone_id:
            # No zone assigned - neutral cost
            return 1.0

        interceptor_zone = self.state.engagement_zones.get(interceptor_zone_id)
        if not interceptor_zone or not interceptor_zone.active:
            return 1.0

        # Check if target is in interceptor's zone
        if interceptor_zone.contains_point(target_position):
            # Target in our zone - reduce cost (preferred)
            return 0.5

        # Check if target is in another zone
        target_zone = self.find_zone_for_point(target_position)
        if target_zone and target_zone.zone_id != interceptor_zone_id:
            # Target in another zone - increase cost (discouraged)
            return 2.0

        # Target not in any zone - normal cost
        return 1.0

    def should_request_handoff(
        self,
        interceptor_id: str,
        target_id: str,
        target_position: Vec3,
        current_time: float
    ) -> Optional[tuple[str, HandoffReason]]:
        """
        Check if interceptor should request handoff for target.

        Returns (suggested_interceptor_id, reason) or None.
        """
        interceptor_zone_id = self.state.interceptor_zones.get(interceptor_id)
        target_zone = self.find_zone_for_point(target_position)

        # If target moved to a different zone
        if target_zone and interceptor_zone_id != target_zone.zone_id:
            # Find an interceptor in the target's zone
            for candidate_id in target_zone.assigned_interceptors:
                if candidate_id != interceptor_id:
                    return (candidate_id, HandoffReason.ZONE_BOUNDARY)

        return None

    # ================== State Access ==================

    def get_state(self) -> CooperativeState:
        """Get current cooperative state."""
        return self.state

    def update(self, current_time: float):
        """
        Periodic update - cleanup expired requests, auto-execute approved handoffs.
        Called each simulation tick.
        """
        self.cleanup_expired_requests(current_time)

        # Auto-execute approved handoffs
        approved = self.get_approved_handoffs()
        for request in approved:
            self.execute_handoff(request.request_id, current_time)
