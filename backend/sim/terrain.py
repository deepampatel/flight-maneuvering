"""
Terrain & 3D Environment Module

This module provides terrain-aware simulation features:
- Digital Elevation Model (DEM) integration
- Procedural terrain generation
- Line-of-sight calculations with terrain masking
- Radar horizon computation
- Urban environment modeling

Physics background:
- Earth curvature affects radar horizon (~4/3 Earth radius for radio waves)
- Terrain masking blocks line-of-sight between entities
- Multipath effects degrade sensor accuracy in urban environments

All features are optional and disabled by default.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math

from .vector import Vec3

# Try to import numpy for terrain operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# Earth parameters
EARTH_RADIUS = 6371000.0  # meters
EFFECTIVE_EARTH_RADIUS = EARTH_RADIUS * 4 / 3  # For radar horizon (atmospheric refraction)


@dataclass
class TerrainConfig:
    """
    Configuration for terrain model.
    """
    # DEM settings
    dem_file: Optional[str] = None           # Path to GeoTIFF or similar
    resolution: float = 30.0                  # meters per grid cell
    bounds: Tuple[float, float, float, float] = (0, 0, 10000, 10000)  # xmin, ymin, xmax, ymax

    # Features
    enable_masking: bool = True               # Use terrain for LOS blocking
    enable_radar_horizon: bool = True         # Include Earth curvature
    samples_per_los: int = 20                 # Samples for LOS raycast

    # Procedural terrain
    procedural_seed: int = 42
    procedural_amplitude: float = 500.0       # meters height variation
    procedural_frequency: float = 0.001       # spatial frequency

    def to_dict(self) -> dict:
        return {
            "dem_file": self.dem_file,
            "resolution": self.resolution,
            "bounds": list(self.bounds),
            "enable_masking": self.enable_masking,
            "enable_radar_horizon": self.enable_radar_horizon,
            "samples_per_los": self.samples_per_los,
            "procedural_seed": self.procedural_seed,
            "procedural_amplitude": self.procedural_amplitude,
            "procedural_frequency": self.procedural_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TerrainConfig":
        bounds = data.get("bounds", [0, 0, 10000, 10000])
        return cls(
            dem_file=data.get("dem_file"),
            resolution=data.get("resolution", 30.0),
            bounds=tuple(bounds) if isinstance(bounds, list) else bounds,
            enable_masking=data.get("enable_masking", True),
            enable_radar_horizon=data.get("enable_radar_horizon", True),
            samples_per_los=data.get("samples_per_los", 20),
            procedural_seed=data.get("procedural_seed", 42),
            procedural_amplitude=data.get("procedural_amplitude", 500.0),
            procedural_frequency=data.get("procedural_frequency", 0.001),
        )


class TerrainModel:
    """
    Digital Elevation Model for terrain-aware simulation.

    Features:
    - Load DEM from file (GeoTIFF) or generate procedurally
    - Query elevation at any point (bilinear interpolation)
    - Line-of-sight checking with terrain masking
    - Radar horizon calculations

    Usage:
        terrain = TerrainModel(config)
        terrain.generate_procedural()  # or terrain.load_dem(filepath)

        # Query elevation
        height = terrain.get_elevation(x, y)

        # Check LOS
        visible = terrain.is_line_of_sight_clear(pos1, pos2)
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()
        self.elevation_grid: Optional[any] = None  # numpy array
        self.grid_width: int = 0
        self.grid_height: int = 0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if terrain data is loaded."""
        return self._loaded and self.elevation_grid is not None

    def load_dem(self, filepath: str) -> bool:
        """
        Load elevation data from file.

        Supports basic numpy .npy files. For GeoTIFF, would need rasterio.
        """
        if not NUMPY_AVAILABLE:
            return False

        try:
            # Try loading as numpy array
            if filepath.endswith('.npy'):
                self.elevation_grid = np.load(filepath)
                self.grid_height, self.grid_width = self.elevation_grid.shape
                self._loaded = True
                return True

            # For GeoTIFF support, would use rasterio:
            # import rasterio
            # with rasterio.open(filepath) as src:
            #     self.elevation_grid = src.read(1)
            #     ...

            return False
        except Exception:
            return False

    def generate_procedural(self, seed: Optional[int] = None) -> None:
        """
        Generate procedural terrain using Perlin-like noise.

        Uses simple value noise for terrain generation.
        """
        if not NUMPY_AVAILABLE:
            # Fallback: flat terrain
            self._loaded = True
            return

        seed = seed or self.config.procedural_seed
        np.random.seed(seed)

        # Grid dimensions from bounds
        xmin, ymin, xmax, ymax = self.config.bounds
        self.grid_width = int((xmax - xmin) / self.config.resolution)
        self.grid_height = int((ymax - ymin) / self.config.resolution)

        # Generate multi-octave noise
        self.elevation_grid = np.zeros((self.grid_height, self.grid_width))

        # Multiple octaves of noise
        for octave in range(4):
            freq = self.config.procedural_frequency * (2 ** octave)
            amp = self.config.procedural_amplitude / (2 ** octave)

            # Create noise at this octave
            noise = self._generate_noise_octave(freq, seed + octave)
            self.elevation_grid += noise * amp

        # Clamp to reasonable values
        self.elevation_grid = np.clip(self.elevation_grid, 0, self.config.procedural_amplitude * 2)

        self._loaded = True

    def _generate_noise_octave(self, frequency: float, seed: int) -> any:
        """Generate a single octave of value noise."""
        if not NUMPY_AVAILABLE:
            return None

        np.random.seed(seed)

        # Coarse random grid
        coarse_w = max(4, int(self.grid_width * frequency * self.config.resolution))
        coarse_h = max(4, int(self.grid_height * frequency * self.config.resolution))

        coarse = np.random.rand(coarse_h, coarse_w)

        # Upsample with interpolation (simple bilinear)
        from scipy import ndimage
        zoom_x = self.grid_width / coarse_w
        zoom_y = self.grid_height / coarse_h

        try:
            upsampled = ndimage.zoom(coarse, (zoom_y, zoom_x), order=1)
            # Trim to exact size
            return upsampled[:self.grid_height, :self.grid_width]
        except Exception:
            # Fallback to zeros
            return np.zeros((self.grid_height, self.grid_width))

    def get_elevation(self, x: float, y: float) -> float:
        """
        Get terrain elevation at world coordinates.

        Uses bilinear interpolation for smooth results.
        """
        if not self.is_loaded or self.elevation_grid is None:
            return 0.0

        # Convert world coordinates to grid indices
        xmin, ymin, xmax, ymax = self.config.bounds

        # Clamp to bounds
        x = max(xmin, min(xmax, x))
        y = max(ymin, min(ymax, y))

        # Grid coordinates (float)
        gx = (x - xmin) / self.config.resolution
        gy = (y - ymin) / self.config.resolution

        # Clamp to grid
        gx = max(0, min(self.grid_width - 1.001, gx))
        gy = max(0, min(self.grid_height - 1.001, gy))

        # Bilinear interpolation
        x0 = int(gx)
        y0 = int(gy)
        x1 = min(x0 + 1, self.grid_width - 1)
        y1 = min(y0 + 1, self.grid_height - 1)

        fx = gx - x0
        fy = gy - y0

        if NUMPY_AVAILABLE and self.elevation_grid is not None:
            # Sample four corners
            v00 = self.elevation_grid[y0, x0]
            v10 = self.elevation_grid[y0, x1]
            v01 = self.elevation_grid[y1, x0]
            v11 = self.elevation_grid[y1, x1]

            # Bilinear blend
            v0 = v00 * (1 - fx) + v10 * fx
            v1 = v01 * (1 - fx) + v11 * fx
            return float(v0 * (1 - fy) + v1 * fy)

        return 0.0

    def get_normal(self, x: float, y: float) -> Vec3:
        """
        Get terrain surface normal at world coordinates.

        Computed from local gradient.
        """
        if not self.is_loaded:
            return Vec3(0, 0, 1)  # Default: flat, pointing up

        # Sample nearby points for gradient
        dx = self.config.resolution
        h_center = self.get_elevation(x, y)
        h_px = self.get_elevation(x + dx, y)
        h_py = self.get_elevation(x, y + dx)

        # Gradient
        dzdx = (h_px - h_center) / dx
        dzdy = (h_py - h_center) / dx

        # Normal from gradient
        normal = Vec3(-dzdx, -dzdy, 1.0)
        return normal.normalized()

    def is_line_of_sight_clear(
        self,
        pos1: Vec3,
        pos2: Vec3,
        samples: Optional[int] = None
    ) -> bool:
        """
        Check if line of sight is blocked by terrain.

        Samples points along the LOS ray and checks against terrain.
        """
        if not self.config.enable_masking or not self.is_loaded:
            return True

        samples = samples or self.config.samples_per_los

        for i in range(1, samples):
            t = i / samples

            # Interpolate position along ray
            px = pos1.x + (pos2.x - pos1.x) * t
            py = pos1.y + (pos2.y - pos1.y) * t
            pz = pos1.z + (pos2.z - pos1.z) * t

            # Get terrain height at this point
            terrain_z = self.get_elevation(px, py)

            # Account for Earth curvature if enabled
            if self.config.enable_radar_horizon:
                dist = math.sqrt((px - pos1.x)**2 + (py - pos1.y)**2)
                curvature_drop = (dist ** 2) / (2 * EFFECTIVE_EARTH_RADIUS)
                terrain_z += curvature_drop

            # Check if ray is below terrain
            if pz < terrain_z:
                return False

        return True

    def get_terrain_mask_angle(
        self,
        observer: Vec3,
        direction: Vec3,
        max_range: float = 10000.0
    ) -> float:
        """
        Get the elevation angle at which terrain blocks view.

        Returns the minimum elevation angle that clears all terrain
        in the given direction.

        Returns:
            Mask angle in degrees (0 = horizon, 90 = straight up)
        """
        if not self.is_loaded:
            return 0.0

        # Normalize direction (horizontal component)
        dir_h = Vec3(direction.x, direction.y, 0)
        if dir_h.magnitude() < 0.01:
            return 0.0
        dir_h = dir_h.normalized()

        max_angle = 0.0
        num_samples = 50

        for i in range(1, num_samples + 1):
            dist = (i / num_samples) * max_range

            # Point on ground at this distance
            px = observer.x + dir_h.x * dist
            py = observer.y + dir_h.y * dist

            terrain_z = self.get_elevation(px, py)

            # Account for Earth curvature
            if self.config.enable_radar_horizon:
                curvature_drop = (dist ** 2) / (2 * EFFECTIVE_EARTH_RADIUS)
                terrain_z += curvature_drop

            # Elevation angle to this terrain point
            height_diff = terrain_z - observer.z
            angle_rad = math.atan2(height_diff, dist)
            angle_deg = math.degrees(angle_rad)

            max_angle = max(max_angle, angle_deg)

        return max_angle

    def compute_radar_horizon(
        self,
        observer_height: float,
        target_height: float = 0.0
    ) -> float:
        """
        Compute radar horizon distance.

        Uses 4/3 Earth radius model for atmospheric refraction.

        Args:
            observer_height: Observer altitude in meters
            target_height: Target altitude in meters

        Returns:
            Maximum detection range in meters (flat Earth approximation)
        """
        # Radar horizon formula: d = sqrt(2 * R * h)
        # where R = 4/3 * Earth radius (atmospheric refraction)

        h_obs = max(0, observer_height)
        h_tgt = max(0, target_height)

        d_obs = math.sqrt(2 * EFFECTIVE_EARTH_RADIUS * h_obs)
        d_tgt = math.sqrt(2 * EFFECTIVE_EARTH_RADIUS * h_tgt)

        return d_obs + d_tgt

    def get_heightmap_data(
        self,
        xmin: Optional[float] = None,
        ymin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymax: Optional[float] = None,
        max_resolution: int = 100
    ) -> dict:
        """
        Get heightmap data for visualization.

        Returns a downsampled heightmap for rendering.
        """
        if not self.is_loaded or not NUMPY_AVAILABLE or self.elevation_grid is None:
            return {
                "width": 0,
                "height": 0,
                "data": [],
                "bounds": list(self.config.bounds),
                "min_elevation": 0,
                "max_elevation": 0,
            }

        # Use full bounds if not specified
        bounds = self.config.bounds
        xmin = xmin or bounds[0]
        ymin = ymin or bounds[1]
        xmax = xmax or bounds[2]
        ymax = ymax or bounds[3]

        # Downsample if needed
        step_x = max(1, self.grid_width // max_resolution)
        step_y = max(1, self.grid_height // max_resolution)

        downsampled = self.elevation_grid[::step_y, ::step_x]

        return {
            "width": downsampled.shape[1],
            "height": downsampled.shape[0],
            "data": downsampled.flatten().tolist(),
            "bounds": [xmin, ymin, xmax, ymax],
            "min_elevation": float(downsampled.min()),
            "max_elevation": float(downsampled.max()),
            "resolution": self.config.resolution * max(step_x, step_y),
        }


@dataclass
class Building:
    """Simple building model for urban environments."""
    center: Vec3
    width: float   # X dimension
    depth: float   # Y dimension
    height: float  # Z dimension
    rotation: float = 0.0  # Degrees

    def to_dict(self) -> dict:
        return {
            "center": self.center.to_dict(),
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "rotation": self.rotation,
        }


class UrbanEnvironment:
    """
    Urban environment model for building-induced effects.

    Features:
    - Building placement
    - Ray-box intersection for LOS blocking
    - Multipath degradation estimation
    """

    def __init__(self):
        self.buildings: List[Building] = []

    def add_building(self, building: Building) -> None:
        """Add a building to the environment."""
        self.buildings.append(building)

    def clear_buildings(self) -> None:
        """Remove all buildings."""
        self.buildings.clear()

    def is_blocked_by_building(self, pos1: Vec3, pos2: Vec3) -> bool:
        """
        Check if line of sight is blocked by any building.

        Uses ray-AABB intersection.
        """
        for building in self.buildings:
            if self._ray_intersects_building(pos1, pos2, building):
                return True
        return False

    def _ray_intersects_building(self, pos1: Vec3, pos2: Vec3, building: Building) -> bool:
        """
        Ray-AABB intersection test.

        Simple slab method for axis-aligned bounding box.
        """
        # Building bounds (axis-aligned, ignoring rotation for simplicity)
        half_w = building.width / 2
        half_d = building.depth / 2

        bmin = Vec3(
            building.center.x - half_w,
            building.center.y - half_d,
            building.center.z
        )
        bmax = Vec3(
            building.center.x + half_w,
            building.center.y + half_d,
            building.center.z + building.height
        )

        # Ray direction
        direction = pos2 - pos1
        length = direction.magnitude()
        if length < 0.001:
            return False

        direction = direction / length

        # Slab intersection
        tmin = 0.0
        tmax = length

        for axis in ['x', 'y', 'z']:
            p1 = getattr(pos1, axis)
            d = getattr(direction, axis)
            bmin_a = getattr(bmin, axis)
            bmax_a = getattr(bmax, axis)

            if abs(d) < 1e-8:
                # Ray parallel to slab
                if p1 < bmin_a or p1 > bmax_a:
                    return False
            else:
                t1 = (bmin_a - p1) / d
                t2 = (bmax_a - p1) / d

                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return False

        return True

    def get_multipath_factor(
        self,
        transmitter: Vec3,
        receiver: Vec3
    ) -> float:
        """
        Estimate multipath degradation factor.

        Returns a value 0-1 where 1 = clear, 0 = heavily degraded.
        """
        # Count nearby buildings that could cause reflections
        mid_point = (transmitter + receiver) / 2
        nearby_buildings = 0

        for building in self.buildings:
            dist = (building.center - mid_point).magnitude()
            if dist < 500:  # Within 500m
                nearby_buildings += 1

        # Simple degradation model
        if nearby_buildings == 0:
            return 1.0
        elif nearby_buildings <= 2:
            return 0.8
        elif nearby_buildings <= 5:
            return 0.6
        else:
            return 0.4

    def to_dict(self) -> dict:
        return {
            "buildings": [b.to_dict() for b in self.buildings],
            "count": len(self.buildings),
        }
