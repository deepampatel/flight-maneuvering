"""
3D Vector Math - The Foundation of Everything

In simulation/robotics/games, EVERYTHING is vectors:
- Position: where is it? (x, y, z) meters
- Velocity: how fast and which direction? (vx, vy, vz) m/s
- Acceleration: how is velocity changing? (ax, ay, az) m/s²

Key insight: We use NumPy arrays for speed, but wrap them for clarity.
Real systems (like Anduril's) use similar abstractions.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Vec3:
    """
    A 3D vector. Immutable-ish for safety.

    Coordinate system (NED - North-East-Down, common in aerospace):
    - x: North (positive = forward)
    - y: East (positive = right)
    - z: Down (positive = below horizon)

    For our MVP, we'll use a simpler ENU (East-North-Up) which is
    more intuitive for visualization:
    - x: East
    - y: North
    - z: Up
    """
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self * scalar

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        """Dot product: measures how aligned two vectors are."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        """Cross product: gives vector perpendicular to both inputs."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        """Length of the vector (Euclidean norm)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Vec3:
        """Unit vector (same direction, length = 1)."""
        mag = self.magnitude()
        if mag < 1e-10:  # Avoid division by zero
            return Vec3(0, 0, 0)
        return self / mag

    def distance_to(self, other: Vec3) -> float:
        """Distance between two points."""
        return (self - other).magnitude()

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for math operations."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vec3":
        """Create from numpy array."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def zero(cls) -> "Vec3":
        """Origin / no movement."""
        return cls(0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        """For JSON serialization."""
        return {"x": self.x, "y": self.y, "z": self.z}

    def __repr__(self) -> str:
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
