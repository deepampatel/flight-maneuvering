"""
Swarm Tactics & Coordination Module

This module provides swarm behavior control for multi-agent scenarios:
- Formation control (V-formation, line abreast, echelon, etc.)
- Boids-inspired steering (separation, alignment, cohesion)
- Tactical maneuvers (saturation, pincer, defensive sphere)

Physics background:
- Reynolds flocking rules for emergent swarm behavior
- Formation slots computed relative to leader position/heading
- Hungarian algorithm for optimal slot assignment

All features are optional and disabled by default.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

from .vector import Vec3
from .entities import Entity


class FormationType(str, Enum):
    """Available formation patterns for swarm control."""
    LINE_ABREAST = "line_abreast"    # Side by side
    ECHELON_RIGHT = "echelon_right"  # Diagonal right
    ECHELON_LEFT = "echelon_left"    # Diagonal left
    V_FORMATION = "v_formation"       # V shape (geese)
    WEDGE = "wedge"                   # Inverted V
    TRAIL = "trail"                   # Single file behind leader
    DIAMOND = "diamond"               # Diamond pattern
    SWARM = "swarm"                   # No fixed formation (pure Boids)


@dataclass
class SwarmConfig:
    """
    Configuration for swarm behavior.

    All weights default to reasonable values for realistic behavior.
    """
    # Formation settings
    formation: FormationType = FormationType.LINE_ABREAST
    spacing: float = 200.0            # meters between agents
    formation_stiffness: float = 0.5  # How strictly to maintain formation (0-1)

    # Boids behavior weights
    separation_weight: float = 1.5    # Avoid crowding neighbors
    alignment_weight: float = 1.0     # Match neighbor velocity
    cohesion_weight: float = 1.0      # Move toward group center
    leader_follow_weight: float = 2.0 # Follow formation leader

    # Collision avoidance
    enable_collision_avoidance: bool = True
    collision_radius: float = 50.0    # meters - minimum separation
    avoidance_strength: float = 5.0   # Force multiplier for collision avoidance

    # Neighbor detection
    neighbor_radius: float = 500.0    # meters - range for Boids calculations
    neighbor_fov: float = 270.0       # degrees - field of view for neighbors

    # Steering limits
    max_steering_accel: float = 30.0  # m/s² - max swarm steering force

    def to_dict(self) -> dict:
        return {
            "formation": self.formation.value,
            "spacing": self.spacing,
            "formation_stiffness": self.formation_stiffness,
            "separation_weight": self.separation_weight,
            "alignment_weight": self.alignment_weight,
            "cohesion_weight": self.cohesion_weight,
            "leader_follow_weight": self.leader_follow_weight,
            "enable_collision_avoidance": self.enable_collision_avoidance,
            "collision_radius": self.collision_radius,
            "avoidance_strength": self.avoidance_strength,
            "neighbor_radius": self.neighbor_radius,
            "neighbor_fov": self.neighbor_fov,
            "max_steering_accel": self.max_steering_accel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SwarmConfig":
        formation = data.get("formation", "line_abreast")
        if isinstance(formation, str):
            formation = FormationType(formation)
        return cls(
            formation=formation,
            spacing=data.get("spacing", 200.0),
            formation_stiffness=data.get("formation_stiffness", 0.5),
            separation_weight=data.get("separation_weight", 1.5),
            alignment_weight=data.get("alignment_weight", 1.0),
            cohesion_weight=data.get("cohesion_weight", 1.0),
            leader_follow_weight=data.get("leader_follow_weight", 2.0),
            enable_collision_avoidance=data.get("enable_collision_avoidance", True),
            collision_radius=data.get("collision_radius", 50.0),
            avoidance_strength=data.get("avoidance_strength", 5.0),
            neighbor_radius=data.get("neighbor_radius", 500.0),
            neighbor_fov=data.get("neighbor_fov", 270.0),
            max_steering_accel=data.get("max_steering_accel", 30.0),
        )


@dataclass
class SwarmState:
    """
    Runtime state for swarm coordination.
    """
    leader_id: Optional[str] = None
    formation_slots: Dict[str, int] = field(default_factory=dict)      # agent_id -> slot_index
    slot_positions: Dict[int, Vec3] = field(default_factory=dict)      # slot_index -> world position
    neighbor_cache: Dict[str, List[str]] = field(default_factory=dict) # agent_id -> neighbor ids

    # Metrics
    formation_error: float = 0.0      # Average deviation from ideal positions
    cohesion_metric: float = 0.0      # Group compactness measure

    def clear_cache(self) -> None:
        """Clear per-tick caches."""
        self.neighbor_cache.clear()


class SwarmController:
    """
    Controls swarm behavior using Boids algorithm with formation overlays.

    Features:
    - Reynolds flocking (separation, alignment, cohesion)
    - Formation control with slot assignment
    - Collision avoidance
    - Leader following

    Usage:
        swarm = SwarmController(config)
        swarm.set_leader(leader_id)

        # Each tick:
        swarm.update(agents, dt)
        for agent in agents:
            steering = swarm.compute_steering(agent, agents)
            agent.acceleration += steering
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.state = SwarmState()

    def set_leader(self, leader_id: str) -> None:
        """Set the formation leader."""
        self.state.leader_id = leader_id

    def set_formation(self, formation: str) -> None:
        """Set the formation type by name."""
        try:
            self.config.formation = FormationType(formation)
        except ValueError:
            pass  # Keep current formation if invalid

    def update(self, agents: List[Entity], dt: float) -> None:
        """
        Update swarm state (call once per tick).

        - Computes formation slot positions
        - Assigns agents to slots
        - Caches neighbor relationships
        """
        self.state.clear_cache()

        if not agents:
            return

        # Auto-select leader if not set
        if self.state.leader_id is None or not any(a.id == self.state.leader_id for a in agents):
            self.state.leader_id = agents[0].id

        # Find leader
        leader = next((a for a in agents if a.id == self.state.leader_id), agents[0])

        # Compute formation slot positions in world coordinates
        self._compute_formation_slots(leader, len(agents))

        # Assign agents to slots (optimal assignment)
        self._assign_slots(agents)

        # Cache neighbors for each agent
        for agent in agents:
            self.state.neighbor_cache[agent.id] = self._find_neighbors(agent, agents)

        # Update metrics
        self._update_metrics(agents)

    def _compute_formation_slots(self, leader: Entity, num_agents: int) -> None:
        """Compute world positions for each formation slot."""
        formation = self.config.formation
        spacing = self.config.spacing

        # Get leader heading (direction of velocity)
        if leader.velocity.magnitude() > 1.0:
            heading = math.atan2(leader.velocity.y, leader.velocity.x)
        else:
            heading = 0.0

        # Compute relative slot positions based on formation type
        relative_positions = self._get_formation_pattern(formation, num_agents, spacing)

        # Transform to world coordinates
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        self.state.slot_positions.clear()
        for i, rel_pos in enumerate(relative_positions):
            # Rotate relative position by heading
            world_x = leader.position.x + rel_pos.x * cos_h - rel_pos.y * sin_h
            world_y = leader.position.y + rel_pos.x * sin_h + rel_pos.y * cos_h
            world_z = leader.position.z + rel_pos.z

            self.state.slot_positions[i] = Vec3(world_x, world_y, world_z)

    def _get_formation_pattern(self, formation: FormationType, n: int, spacing: float) -> List[Vec3]:
        """
        Get relative positions for formation slots.

        Positions are relative to leader (slot 0 = leader position).
        X = forward (positive = ahead), Y = lateral (positive = right), Z = vertical
        """
        positions = [Vec3.zero()]  # Leader at origin

        if n <= 1:
            return positions

        if formation == FormationType.LINE_ABREAST:
            # Side by side
            for i in range(1, n):
                side = 1 if i % 2 == 1 else -1
                offset = ((i + 1) // 2) * spacing * side
                positions.append(Vec3(0, offset, 0))

        elif formation == FormationType.ECHELON_RIGHT:
            # Diagonal trailing right
            for i in range(1, n):
                positions.append(Vec3(-i * spacing * 0.7, i * spacing * 0.7, 0))

        elif formation == FormationType.ECHELON_LEFT:
            # Diagonal trailing left
            for i in range(1, n):
                positions.append(Vec3(-i * spacing * 0.7, -i * spacing * 0.7, 0))

        elif formation == FormationType.V_FORMATION:
            # V shape (like geese)
            for i in range(1, n):
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                positions.append(Vec3(-rank * spacing * 0.7, rank * spacing * 0.7 * side, 0))

        elif formation == FormationType.WEDGE:
            # Inverted V (leader at front)
            for i in range(1, n):
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                positions.append(Vec3(-rank * spacing * 0.7, rank * spacing * 0.7 * side, 0))

        elif formation == FormationType.TRAIL:
            # Single file
            for i in range(1, n):
                positions.append(Vec3(-i * spacing, 0, 0))

        elif formation == FormationType.DIAMOND:
            # Diamond pattern
            if n >= 2:
                positions.append(Vec3(0, spacing, 0))  # Right
            if n >= 3:
                positions.append(Vec3(0, -spacing, 0))  # Left
            if n >= 4:
                positions.append(Vec3(-spacing, 0, 0))  # Trail
            for i in range(4, n):
                # Additional agents form outer ring
                angle = (i - 4) * (2 * math.pi / (n - 4)) if n > 4 else 0
                positions.append(Vec3(
                    -spacing * 0.5 + spacing * 1.5 * math.cos(angle),
                    spacing * 1.5 * math.sin(angle),
                    0
                ))

        else:  # SWARM - no fixed formation
            # Just spread agents in a loose cluster
            for i in range(1, n):
                angle = i * (2 * math.pi / (n - 1)) if n > 1 else 0
                radius = spacing * 0.5
                positions.append(Vec3(
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0
                ))

        return positions

    def _assign_slots(self, agents: List[Entity]) -> None:
        """
        Assign agents to formation slots using greedy nearest approach.

        For small swarms, greedy is sufficient. For larger swarms,
        Hungarian algorithm could be used for optimal assignment.
        """
        if not agents:
            return

        self.state.formation_slots.clear()
        assigned_slots = set()

        # Leader always gets slot 0
        leader = next((a for a in agents if a.id == self.state.leader_id), None)
        if leader:
            self.state.formation_slots[leader.id] = 0
            assigned_slots.add(0)

        # Greedily assign remaining agents to nearest unassigned slots
        for agent in agents:
            if agent.id in self.state.formation_slots:
                continue

            best_slot = -1
            best_dist = float('inf')

            for slot_idx, slot_pos in self.state.slot_positions.items():
                if slot_idx in assigned_slots:
                    continue

                dist = (agent.position - slot_pos).magnitude()
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot_idx

            if best_slot >= 0:
                self.state.formation_slots[agent.id] = best_slot
                assigned_slots.add(best_slot)

    def _find_neighbors(self, agent: Entity, agents: List[Entity]) -> List[str]:
        """Find neighbors within radius and FOV."""
        neighbors = []

        for other in agents:
            if other.id == agent.id:
                continue

            rel_pos = other.position - agent.position
            distance = rel_pos.magnitude()

            if distance > self.config.neighbor_radius:
                continue

            # Check FOV (if agent has velocity)
            if agent.velocity.magnitude() > 1.0 and self.config.neighbor_fov < 360:
                agent_heading = agent.velocity.normalized()
                to_neighbor = rel_pos.normalized()
                dot = agent_heading.x * to_neighbor.x + agent_heading.y * to_neighbor.y
                angle_deg = math.degrees(math.acos(max(-1, min(1, dot))))

                if angle_deg > self.config.neighbor_fov / 2:
                    continue

            neighbors.append(other.id)

        return neighbors

    def _update_metrics(self, agents: List[Entity]) -> None:
        """Update swarm health metrics."""
        if not agents:
            return

        # Formation error: average deviation from ideal slot positions
        total_error = 0.0
        count = 0
        for agent in agents:
            slot_idx = self.state.formation_slots.get(agent.id)
            if slot_idx is not None and slot_idx in self.state.slot_positions:
                ideal_pos = self.state.slot_positions[slot_idx]
                error = (agent.position - ideal_pos).magnitude()
                total_error += error
                count += 1

        self.state.formation_error = total_error / count if count > 0 else 0.0

        # Cohesion: inverse of average distance to centroid
        if len(agents) > 1:
            centroid = Vec3.zero()
            for a in agents:
                centroid = centroid + a.position
            centroid = centroid / len(agents)

            avg_dist = sum((a.position - centroid).magnitude() for a in agents) / len(agents)
            self.state.cohesion_metric = 1.0 / (1.0 + avg_dist / self.config.spacing)
        else:
            self.state.cohesion_metric = 1.0

    def compute_steering(self, agent: Entity, agents: List[Entity]) -> Vec3:
        """
        Compute swarm steering acceleration for an agent.

        Combines:
        - Separation (avoid crowding)
        - Alignment (match velocity)
        - Cohesion (stay together)
        - Formation (move to slot)
        - Collision avoidance (emergency)
        """
        # Get cached neighbors
        neighbor_ids = self.state.neighbor_cache.get(agent.id, [])
        neighbors = [a for a in agents if a.id in neighbor_ids]

        # Compute individual steering components
        separation = self._compute_separation(agent, neighbors)
        alignment = self._compute_alignment(agent, neighbors)
        cohesion = self._compute_cohesion(agent, neighbors)
        formation = self._compute_formation_steering(agent)

        # Combine with weights
        steering = (
            separation * self.config.separation_weight +
            alignment * self.config.alignment_weight +
            cohesion * self.config.cohesion_weight +
            formation * self.config.leader_follow_weight * self.config.formation_stiffness
        )

        # Add collision avoidance (high priority)
        if self.config.enable_collision_avoidance:
            avoidance = self._compute_collision_avoidance(agent, agents)
            steering = steering + avoidance * self.config.avoidance_strength

        # Limit steering magnitude
        mag = steering.magnitude()
        if mag > self.config.max_steering_accel:
            steering = steering.normalized() * self.config.max_steering_accel

        return steering

    def _compute_separation(self, agent: Entity, neighbors: List[Entity]) -> Vec3:
        """
        Steer to avoid crowding neighbors.

        Force is inversely proportional to distance.
        """
        if not neighbors:
            return Vec3.zero()

        steering = Vec3.zero()

        for neighbor in neighbors:
            rel_pos = agent.position - neighbor.position
            distance = rel_pos.magnitude()

            if distance < 0.1:
                distance = 0.1

            # Stronger repulsion when closer
            force = rel_pos.normalized() / distance
            steering = steering + force

        if len(neighbors) > 0:
            steering = steering / len(neighbors)

        return steering * 10.0  # Scale factor

    def _compute_alignment(self, agent: Entity, neighbors: List[Entity]) -> Vec3:
        """
        Steer towards average heading of neighbors.

        Helps maintain coherent group direction.
        """
        if not neighbors:
            return Vec3.zero()

        avg_velocity = Vec3.zero()
        for neighbor in neighbors:
            avg_velocity = avg_velocity + neighbor.velocity

        avg_velocity = avg_velocity / len(neighbors)

        # Steer towards average velocity
        desired = avg_velocity - agent.velocity
        return desired * 0.1  # Scale factor

    def _compute_cohesion(self, agent: Entity, neighbors: List[Entity]) -> Vec3:
        """
        Steer towards center of mass of neighbors.

        Keeps the group together.
        """
        if not neighbors:
            return Vec3.zero()

        center = Vec3.zero()
        for neighbor in neighbors:
            center = center + neighbor.position

        center = center / len(neighbors)

        # Steer towards center
        desired = center - agent.position
        return desired * 0.01  # Scale factor

    def _compute_formation_steering(self, agent: Entity) -> Vec3:
        """
        Steer towards assigned formation slot.
        """
        slot_idx = self.state.formation_slots.get(agent.id)
        if slot_idx is None or slot_idx not in self.state.slot_positions:
            return Vec3.zero()

        target_pos = self.state.slot_positions[slot_idx]
        rel_pos = target_pos - agent.position
        distance = rel_pos.magnitude()

        if distance < 1.0:
            return Vec3.zero()

        # Proportional steering towards slot
        # Stronger when farther from slot
        strength = min(distance / self.config.spacing, 2.0)
        return rel_pos.normalized() * strength * 5.0

    def _compute_collision_avoidance(self, agent: Entity, agents: List[Entity]) -> Vec3:
        """
        Emergency collision avoidance.

        Strong repulsive force when too close to another agent.
        """
        steering = Vec3.zero()

        for other in agents:
            if other.id == agent.id:
                continue

            rel_pos = agent.position - other.position
            distance = rel_pos.magnitude()

            if distance < self.config.collision_radius:
                # Strong repulsion inversely proportional to distance
                if distance < 1.0:
                    distance = 1.0

                force_mag = (self.config.collision_radius - distance) / distance
                steering = steering + rel_pos.normalized() * force_mag

        return steering


class SwarmTactics:
    """
    Higher-level tactical maneuvers for swarms.

    These override normal formation behavior with specific attack patterns.
    """

    @staticmethod
    def saturate_target(
        agents: List[Entity],
        target: Entity,
        approach_radius: float = 500.0
    ) -> Dict[str, Vec3]:
        """
        Coordinate swarm to approach target from multiple angles.

        Distributes agents evenly around target for simultaneous arrival.

        Returns:
            Dict mapping agent_id to target position
        """
        if not agents:
            return {}

        n = len(agents)
        target_positions = {}

        for i, agent in enumerate(agents):
            # Compute angle for this agent
            angle = (2 * math.pi * i) / n

            # Position on circle around target
            offset_x = approach_radius * math.cos(angle)
            offset_y = approach_radius * math.sin(angle)

            # Target position is on the approach circle
            target_pos = Vec3(
                target.position.x + offset_x,
                target.position.y + offset_y,
                target.position.z
            )

            target_positions[agent.id] = target_pos

        return target_positions

    @staticmethod
    def pincer_maneuver(
        agents: List[Entity],
        target: Entity,
        flank_angle: float = 60.0,
        approach_distance: float = 1000.0
    ) -> Tuple[Dict[str, Vec3], Dict[str, Vec3]]:
        """
        Split swarm into two groups for flanking attack.

        Returns:
            Tuple of (group_assignments, target_positions)
            group_assignments: agent_id -> "left" or "right"
            target_positions: agent_id -> waypoint position
        """
        if not agents:
            return {}, {}

        # Split agents into two groups
        mid = len(agents) // 2
        left_group = agents[:mid]
        right_group = agents[mid:]

        # Compute flank positions
        # Get target heading (or use default if stationary)
        if target.velocity.magnitude() > 1.0:
            target_heading = math.atan2(target.velocity.y, target.velocity.x)
        else:
            target_heading = 0.0

        flank_rad = math.radians(flank_angle)

        group_assignments = {}
        target_positions = {}

        # Left flank
        left_angle = target_heading + math.pi + flank_rad
        for i, agent in enumerate(left_group):
            group_assignments[agent.id] = "left"
            offset = approach_distance * (1 + i * 0.2)  # Stagger
            target_positions[agent.id] = Vec3(
                target.position.x + offset * math.cos(left_angle),
                target.position.y + offset * math.sin(left_angle),
                target.position.z
            )

        # Right flank
        right_angle = target_heading + math.pi - flank_rad
        for i, agent in enumerate(right_group):
            group_assignments[agent.id] = "right"
            offset = approach_distance * (1 + i * 0.2)
            target_positions[agent.id] = Vec3(
                target.position.x + offset * math.cos(right_angle),
                target.position.y + offset * math.sin(right_angle),
                target.position.z
            )

        return group_assignments, target_positions

    @staticmethod
    def defensive_sphere(
        agents: List[Entity],
        asset: Entity,
        radius: float = 300.0,
        layers: int = 2
    ) -> Dict[str, Vec3]:
        """
        Form protective sphere around high-value asset.

        Distributes agents in spherical shell(s) around asset.

        Returns:
            Dict mapping agent_id to defensive position
        """
        if not agents:
            return {}

        target_positions = {}
        n = len(agents)

        # Distribute across layers
        agents_per_layer = n // layers if layers > 0 else n

        for i, agent in enumerate(agents):
            layer = i // agents_per_layer if agents_per_layer > 0 else 0
            idx_in_layer = i % agents_per_layer if agents_per_layer > 0 else i
            layer_count = min(agents_per_layer, n - layer * agents_per_layer)

            # Spherical coordinates
            layer_radius = radius * (1 + layer * 0.5)

            # Use golden ratio for even distribution
            golden_angle = math.pi * (3 - math.sqrt(5))
            theta = golden_angle * idx_in_layer
            phi = math.acos(1 - 2 * (idx_in_layer + 0.5) / layer_count) if layer_count > 1 else math.pi / 2

            # Convert to Cartesian
            x = layer_radius * math.sin(phi) * math.cos(theta)
            y = layer_radius * math.sin(phi) * math.sin(theta)
            z = layer_radius * math.cos(phi)

            target_positions[agent.id] = Vec3(
                asset.position.x + x,
                asset.position.y + y,
                asset.position.z + z
            )

        return target_positions


def get_available_formations() -> List[dict]:
    """Get list of available formation types with descriptions."""
    return [
        {"id": "line_abreast", "name": "Line Abreast", "description": "Side by side formation"},
        {"id": "echelon_right", "name": "Echelon Right", "description": "Diagonal trailing to the right"},
        {"id": "echelon_left", "name": "Echelon Left", "description": "Diagonal trailing to the left"},
        {"id": "v_formation", "name": "V Formation", "description": "V shape like flying geese"},
        {"id": "wedge", "name": "Wedge", "description": "Inverted V with leader at front"},
        {"id": "trail", "name": "Trail", "description": "Single file behind leader"},
        {"id": "diamond", "name": "Diamond", "description": "Diamond pattern for 4+ agents"},
        {"id": "swarm", "name": "Swarm", "description": "No fixed formation, pure Boids behavior"},
    ]
