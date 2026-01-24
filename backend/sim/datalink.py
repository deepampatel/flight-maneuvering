"""
Communication & Datalink Modeling Module

This module simulates communication constraints in networked systems:
- Bandwidth-limited datalinks
- Message latency and jitter
- Packet loss modeling
- Link jamming effects
- Networked fire control coordination

Physics background:
- Radio propagation: free space path loss, terrain effects
- Communication delays: speed of light + processing latency
- Packet loss: environmental noise, interference, jamming

All features are optional and disabled by default.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import random
import uuid
import math

from .vector import Vec3


class LinkType(str, Enum):
    """Types of communication links."""
    RADIO = "radio"           # Line-of-sight radio
    SATELLITE = "satellite"   # Satellite relay (higher latency)
    LASER = "laser"           # Directional laser (requires LOS)
    TACTICAL = "tactical"     # Short-range tactical datalink


class MessagePriority(str, Enum):
    """Message priority levels for queue management."""
    CRITICAL = "critical"     # Highest priority (weapons release, emergency)
    HIGH = "high"             # Important tactical data
    NORMAL = "normal"         # Standard track updates
    LOW = "low"               # Background/housekeeping


class MessageType(str, Enum):
    """Types of messages that can be sent."""
    TRACK_UPDATE = "track_update"           # Target track data
    ENGAGEMENT_REQUEST = "engagement_request"
    ENGAGEMENT_CONFIRM = "engagement_confirm"
    HANDOFF = "handoff"
    STATUS = "status"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    """
    A message transmitted over the datalink.
    """
    msg_id: str
    sender_id: str
    recipient_id: str              # "*" for broadcast
    msg_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: float               # When message was created
    size_bytes: int
    ttl: float = 5.0               # Time to live in seconds
    hops: int = 0                  # Number of relay hops

    @classmethod
    def create(
        cls,
        sender_id: str,
        recipient_id: str,
        msg_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        timestamp: float = 0.0,
        ttl: float = 5.0
    ) -> "Message":
        """Create a new message with auto-generated ID."""
        # Estimate size based on payload
        size = 64 + len(str(payload)) * 2  # Header + payload estimate

        return cls(
            msg_id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            recipient_id=recipient_id,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
            timestamp=timestamp,
            size_bytes=size,
            ttl=ttl,
        )

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "msg_type": self.msg_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "ttl": self.ttl,
            "hops": self.hops,
        }


@dataclass
class DatalinkConfig:
    """
    Configuration for datalink simulation.
    """
    # Bandwidth settings
    bandwidth_kbps: float = 100.0        # kilobits per second
    bandwidth_window_ms: float = 100.0   # Time window for bandwidth calculation

    # Latency settings
    base_latency_ms: float = 50.0        # Base one-way latency
    latency_jitter_ms: float = 10.0      # Random variation in latency
    distance_latency_factor: float = 0.0  # Additional ms per km distance

    # Reliability settings
    packet_loss_rate: float = 0.01       # Base packet loss probability
    max_range_km: float = 50.0           # Maximum reliable range

    # Jamming settings
    enable_jamming: bool = False
    jam_effectiveness: float = 0.5       # How much jamming increases loss

    # Queue settings
    max_queue_size: int = 100
    priority_queue: bool = True          # Use priority queue

    def to_dict(self) -> dict:
        return {
            "bandwidth_kbps": self.bandwidth_kbps,
            "bandwidth_window_ms": self.bandwidth_window_ms,
            "base_latency_ms": self.base_latency_ms,
            "latency_jitter_ms": self.latency_jitter_ms,
            "distance_latency_factor": self.distance_latency_factor,
            "packet_loss_rate": self.packet_loss_rate,
            "max_range_km": self.max_range_km,
            "enable_jamming": self.enable_jamming,
            "jam_effectiveness": self.jam_effectiveness,
            "max_queue_size": self.max_queue_size,
            "priority_queue": self.priority_queue,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatalinkConfig":
        return cls(
            bandwidth_kbps=data.get("bandwidth_kbps", 100.0),
            bandwidth_window_ms=data.get("bandwidth_window_ms", 100.0),
            base_latency_ms=data.get("base_latency_ms", 50.0),
            latency_jitter_ms=data.get("latency_jitter_ms", 10.0),
            distance_latency_factor=data.get("distance_latency_factor", 0.0),
            packet_loss_rate=data.get("packet_loss_rate", 0.01),
            max_range_km=data.get("max_range_km", 50.0),
            enable_jamming=data.get("enable_jamming", False),
            jam_effectiveness=data.get("jam_effectiveness", 0.5),
            max_queue_size=data.get("max_queue_size", 100),
            priority_queue=data.get("priority_queue", True),
        )


@dataclass
class DatalinkStats:
    """Statistics for datalink performance."""
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_expired: int = 0
    bytes_sent: int = 0
    bytes_delivered: int = 0
    average_latency_ms: float = 0.0
    bandwidth_utilization: float = 0.0
    current_queue_size: int = 0

    def to_dict(self) -> dict:
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "messages_expired": self.messages_expired,
            "bytes_sent": self.bytes_sent,
            "bytes_delivered": self.bytes_delivered,
            "average_latency_ms": self.average_latency_ms,
            "bandwidth_utilization": self.bandwidth_utilization,
            "current_queue_size": self.current_queue_size,
        }


@dataclass
class Jammer:
    """A jamming source that degrades communications."""
    jammer_id: str
    position: Vec3
    power: float           # Relative jamming power (0-1)
    radius: float          # Effective radius in meters
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "jammer_id": self.jammer_id,
            "position": self.position.to_dict(),
            "power": self.power,
            "radius": self.radius,
            "active": self.active,
        }


class DatalinkModel:
    """
    Simulates communication datalinks with realistic constraints.

    Features:
    - Bandwidth-limited transmission
    - Latency with jitter
    - Packet loss (range-dependent + random)
    - Priority queue for critical messages
    - Jamming effects

    Usage:
        datalink = DatalinkModel(config)

        # Send a message
        datalink.send_message(msg, sender_pos, recipient_pos)

        # Update each tick
        delivered = datalink.update(current_time)

        # Process delivered messages
        for msg in delivered:
            handle_message(msg)
    """

    def __init__(self, config: Optional[DatalinkConfig] = None):
        self.config = config or DatalinkConfig()

        # Message queues
        self.outgoing_queue: List[Message] = []
        self.in_flight: List[Tuple[Message, float]] = []  # (message, arrival_time)

        # Statistics
        self.stats = DatalinkStats()
        self._latency_samples: List[float] = []

        # Jamming
        self.jammers: List[Jammer] = []

        # Bandwidth tracking
        self._bytes_this_window: int = 0
        self._window_start_time: float = 0.0

        # Entity positions (for distance calculations)
        self._entity_positions: Dict[str, Vec3] = {}

    def register_entity(self, entity_id: str, position: Vec3) -> None:
        """Register an entity's position for distance calculations."""
        self._entity_positions[entity_id] = position

    def update_entity_position(self, entity_id: str, position: Vec3) -> None:
        """Update an entity's position."""
        self._entity_positions[entity_id] = position

    def add_jammer(self, jammer: Jammer) -> None:
        """Add a jamming source."""
        self.jammers.append(jammer)

    def remove_jammer(self, jammer_id: str) -> None:
        """Remove a jamming source."""
        self.jammers = [j for j in self.jammers if j.jammer_id != jammer_id]

    def send_message(
        self,
        msg: Message,
        sender_pos: Optional[Vec3] = None,
        recipient_pos: Optional[Vec3] = None
    ) -> bool:
        """
        Queue a message for transmission.

        Returns True if message was queued, False if dropped.
        """
        # Get positions from registry if not provided
        if sender_pos is None:
            sender_pos = self._entity_positions.get(msg.sender_id, Vec3.zero())
        if recipient_pos is None and msg.recipient_id != "*":
            recipient_pos = self._entity_positions.get(msg.recipient_id, Vec3.zero())

        # Check queue capacity
        if len(self.outgoing_queue) >= self.config.max_queue_size:
            if self.config.priority_queue:
                # Drop lowest priority message if new one is higher
                lowest = min(self.outgoing_queue, key=lambda m: self._priority_value(m.priority))
                if self._priority_value(msg.priority) > self._priority_value(lowest.priority):
                    self.outgoing_queue.remove(lowest)
                    self.stats.messages_dropped += 1
                else:
                    self.stats.messages_dropped += 1
                    return False
            else:
                self.stats.messages_dropped += 1
                return False

        # Check range (for non-broadcast)
        if msg.recipient_id != "*" and recipient_pos is not None:
            distance = (recipient_pos - sender_pos).magnitude()
            if distance > self.config.max_range_km * 1000:
                self.stats.messages_dropped += 1
                return False

        # Add to queue
        if self.config.priority_queue:
            # Insert by priority
            inserted = False
            for i, queued in enumerate(self.outgoing_queue):
                if self._priority_value(msg.priority) > self._priority_value(queued.priority):
                    self.outgoing_queue.insert(i, msg)
                    inserted = True
                    break
            if not inserted:
                self.outgoing_queue.append(msg)
        else:
            self.outgoing_queue.append(msg)

        self.stats.messages_sent += 1
        self.stats.bytes_sent += msg.size_bytes

        return True

    def _priority_value(self, priority: MessagePriority) -> int:
        """Convert priority to numeric value."""
        return {
            MessagePriority.CRITICAL: 4,
            MessagePriority.HIGH: 3,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 1,
        }.get(priority, 2)

    def update(self, current_time: float) -> List[Message]:
        """
        Process message transmission and delivery.

        Call once per simulation tick.

        Returns:
            List of messages delivered this tick
        """
        delivered = []

        # Reset bandwidth window if needed
        window_ms = self.config.bandwidth_window_ms
        if current_time - self._window_start_time > window_ms / 1000:
            self._bytes_this_window = 0
            self._window_start_time = current_time

        # Calculate available bandwidth this tick
        max_bytes_per_window = int(self.config.bandwidth_kbps * window_ms / 8)  # kbits to bytes
        available_bytes = max_bytes_per_window - self._bytes_this_window

        # Process outgoing queue
        messages_to_send = []
        remaining_queue = []

        for msg in self.outgoing_queue:
            if msg.size_bytes <= available_bytes:
                messages_to_send.append(msg)
                available_bytes -= msg.size_bytes
                self._bytes_this_window += msg.size_bytes
            else:
                remaining_queue.append(msg)

        self.outgoing_queue = remaining_queue

        # Transmit messages (add to in-flight with arrival time)
        for msg in messages_to_send:
            arrival_time = self._compute_arrival_time(msg, current_time)

            # Check for packet loss
            if self._should_drop_message(msg):
                self.stats.messages_dropped += 1
                continue

            self.in_flight.append((msg, arrival_time))

        # Check in-flight messages for delivery
        still_in_flight = []
        for msg, arrival_time in self.in_flight:
            if current_time >= arrival_time:
                # Check TTL
                if current_time - msg.timestamp > msg.ttl:
                    self.stats.messages_expired += 1
                else:
                    delivered.append(msg)
                    self.stats.messages_delivered += 1
                    self.stats.bytes_delivered += msg.size_bytes

                    # Track latency
                    latency = (current_time - msg.timestamp) * 1000  # to ms
                    self._latency_samples.append(latency)
                    if len(self._latency_samples) > 100:
                        self._latency_samples.pop(0)
            else:
                still_in_flight.append((msg, arrival_time))

        self.in_flight = still_in_flight

        # Update statistics
        self._update_stats()

        return delivered

    def _compute_arrival_time(self, msg: Message, current_time: float) -> float:
        """Compute when a message will arrive at its destination."""
        # Base latency
        latency_ms = self.config.base_latency_ms

        # Add jitter
        if self.config.latency_jitter_ms > 0:
            latency_ms += random.gauss(0, self.config.latency_jitter_ms)

        # Add distance-based latency
        if self.config.distance_latency_factor > 0:
            sender_pos = self._entity_positions.get(msg.sender_id)
            recipient_pos = self._entity_positions.get(msg.recipient_id)
            if sender_pos and recipient_pos:
                distance_km = (recipient_pos - sender_pos).magnitude() / 1000
                latency_ms += distance_km * self.config.distance_latency_factor

        # Ensure positive latency
        latency_ms = max(1.0, latency_ms)

        return current_time + latency_ms / 1000

    def _should_drop_message(self, msg: Message) -> bool:
        """Determine if a message should be dropped."""
        drop_prob = self.config.packet_loss_rate

        # Increase drop probability with distance
        sender_pos = self._entity_positions.get(msg.sender_id)
        recipient_pos = self._entity_positions.get(msg.recipient_id)
        if sender_pos and recipient_pos and msg.recipient_id != "*":
            distance_km = (recipient_pos - sender_pos).magnitude() / 1000
            max_range = self.config.max_range_km

            # Quadratic increase in loss as approaching max range
            if distance_km > max_range * 0.7:
                range_factor = (distance_km - max_range * 0.7) / (max_range * 0.3)
                drop_prob += range_factor * 0.3

        # Apply jamming effects
        if self.config.enable_jamming and sender_pos:
            jam_factor = self._compute_jamming_factor(sender_pos)
            drop_prob += jam_factor * self.config.jam_effectiveness

        # Clamp probability
        drop_prob = min(1.0, drop_prob)

        return random.random() < drop_prob

    def _compute_jamming_factor(self, position: Vec3) -> float:
        """Compute jamming effect at a position."""
        total_jamming = 0.0

        for jammer in self.jammers:
            if not jammer.active:
                continue

            distance = (position - jammer.position).magnitude()
            if distance < jammer.radius:
                # Inverse square law
                effectiveness = jammer.power * (1 - (distance / jammer.radius) ** 2)
                total_jamming = max(total_jamming, effectiveness)

        return min(1.0, total_jamming)

    def _update_stats(self) -> None:
        """Update performance statistics."""
        if self._latency_samples:
            self.stats.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

        max_bytes = int(self.config.bandwidth_kbps * self.config.bandwidth_window_ms / 8)
        self.stats.bandwidth_utilization = self._bytes_this_window / max_bytes if max_bytes > 0 else 0

        self.stats.current_queue_size = len(self.outgoing_queue)

    def get_link_quality(self, entity1_id: str, entity2_id: str) -> float:
        """
        Get estimated link quality between two entities.

        Returns 0-1 where 1 = excellent, 0 = no link.
        """
        pos1 = self._entity_positions.get(entity1_id)
        pos2 = self._entity_positions.get(entity2_id)

        if pos1 is None or pos2 is None:
            return 0.0

        distance_km = (pos2 - pos1).magnitude() / 1000

        # Range factor
        if distance_km > self.config.max_range_km:
            return 0.0

        range_quality = 1.0 - (distance_km / self.config.max_range_km) ** 2

        # Jamming factor
        jam_factor = 0.0
        if self.config.enable_jamming:
            mid_point = (pos1 + pos2) / 2
            jam_factor = self._compute_jamming_factor(mid_point)

        quality = range_quality * (1 - jam_factor * self.config.jam_effectiveness)
        return max(0.0, min(1.0, quality))

    def get_stats(self) -> DatalinkStats:
        """Get current statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = DatalinkStats()
        self._latency_samples.clear()


class NetworkedFireControl:
    """
    Coordinates fire control data across multiple platforms.

    Features:
    - Track sharing via datalink
    - Engagement requests and confirmations
    - Weapon-target deconfliction
    """

    def __init__(self, datalink: DatalinkModel):
        self.datalink = datalink
        self.track_database: Dict[str, Dict] = {}  # target_id -> track data
        self.engagements: Dict[str, Dict] = {}     # target_id -> engagement info
        self._pending_requests: List[Dict] = []

    def share_track(
        self,
        sender_id: str,
        target_id: str,
        track_data: Dict,
        timestamp: float
    ) -> None:
        """Broadcast track update to all platforms."""
        msg = Message.create(
            sender_id=sender_id,
            recipient_id="*",  # Broadcast
            msg_type=MessageType.TRACK_UPDATE,
            payload={
                "target_id": target_id,
                "track": track_data,
            },
            priority=MessagePriority.NORMAL,
            timestamp=timestamp,
        )

        sender_pos = self.datalink._entity_positions.get(sender_id)
        self.datalink.send_message(msg, sender_pos)

    def request_engagement(
        self,
        requester_id: str,
        target_id: str,
        shooter_id: str,
        timestamp: float
    ) -> str:
        """
        Request another platform to engage a target.

        Returns request ID for tracking.
        """
        request_id = str(uuid.uuid4())[:8]

        msg = Message.create(
            sender_id=requester_id,
            recipient_id=shooter_id,
            msg_type=MessageType.ENGAGEMENT_REQUEST,
            payload={
                "request_id": request_id,
                "target_id": target_id,
                "requester_id": requester_id,
            },
            priority=MessagePriority.HIGH,
            timestamp=timestamp,
        )

        self.datalink.send_message(msg)

        self._pending_requests.append({
            "request_id": request_id,
            "target_id": target_id,
            "requester_id": requester_id,
            "shooter_id": shooter_id,
            "timestamp": timestamp,
            "status": "pending",
        })

        return request_id

    def confirm_engagement(
        self,
        shooter_id: str,
        request_id: str,
        target_id: str,
        weapon_id: str,
        timestamp: float
    ) -> None:
        """Confirm weapon assignment to prevent fratricide."""
        # Update local engagement tracking
        self.engagements[target_id] = {
            "shooter_id": shooter_id,
            "weapon_id": weapon_id,
            "timestamp": timestamp,
            "confirmed": True,
        }

        # Broadcast confirmation
        msg = Message.create(
            sender_id=shooter_id,
            recipient_id="*",
            msg_type=MessageType.ENGAGEMENT_CONFIRM,
            payload={
                "request_id": request_id,
                "target_id": target_id,
                "weapon_id": weapon_id,
                "shooter_id": shooter_id,
            },
            priority=MessagePriority.CRITICAL,
            timestamp=timestamp,
        )

        self.datalink.send_message(msg)

    def process_messages(self, messages: List[Message]) -> List[Dict]:
        """
        Process incoming datalink messages.

        Returns list of events/actions to take.
        """
        events = []

        for msg in messages:
            if msg.msg_type == MessageType.TRACK_UPDATE:
                target_id = msg.payload.get("target_id")
                track_data = msg.payload.get("track")
                if target_id and track_data:
                    self.track_database[target_id] = {
                        "data": track_data,
                        "source": msg.sender_id,
                        "timestamp": msg.timestamp,
                    }
                    events.append({
                        "type": "track_update",
                        "target_id": target_id,
                        "source": msg.sender_id,
                    })

            elif msg.msg_type == MessageType.ENGAGEMENT_REQUEST:
                events.append({
                    "type": "engagement_request",
                    "request_id": msg.payload.get("request_id"),
                    "target_id": msg.payload.get("target_id"),
                    "requester_id": msg.payload.get("requester_id"),
                })

            elif msg.msg_type == MessageType.ENGAGEMENT_CONFIRM:
                target_id = msg.payload.get("target_id")
                self.engagements[target_id] = {
                    "shooter_id": msg.payload.get("shooter_id"),
                    "weapon_id": msg.payload.get("weapon_id"),
                    "timestamp": msg.timestamp,
                    "confirmed": True,
                }
                events.append({
                    "type": "engagement_confirmed",
                    "target_id": target_id,
                    "shooter_id": msg.payload.get("shooter_id"),
                })

        return events

    def is_target_engaged(self, target_id: str) -> bool:
        """Check if a target is already being engaged."""
        return target_id in self.engagements and self.engagements[target_id].get("confirmed", False)

    def get_shared_tracks(self) -> Dict[str, Dict]:
        """Get all shared track data."""
        return self.track_database.copy()
