/**
 * TerrainMesh - 3D heightmap visualization
 *
 * Renders terrain from backend heightmap data as a displaced PlaneGeometry
 * with vertex colors based on elevation (green → brown → white).
 */

import { useMemo } from 'react';
import * as THREE from 'three';

const SCALE = 0.001; // Match scene scale

interface HeightmapData {
  width: number;
  height: number;
  data: number[];
  bounds: [number, number, number, number]; // xmin, ymin, xmax, ymax
  min_elevation: number;
  max_elevation: number;
  resolution: number;
}

interface TerrainMeshProps {
  heightmap: HeightmapData;
}

// Color gradient for elevation: dark green → olive → brown → gray → white
function elevationColor(t: number): [number, number, number] {
  if (t < 0.25) {
    // Dark green to olive
    const s = t / 0.25;
    return [0.15 + s * 0.2, 0.35 + s * 0.1, 0.1];
  } else if (t < 0.5) {
    // Olive to brown
    const s = (t - 0.25) / 0.25;
    return [0.35 + s * 0.25, 0.45 - s * 0.15, 0.1 + s * 0.05];
  } else if (t < 0.75) {
    // Brown to gray
    const s = (t - 0.5) / 0.25;
    return [0.6 - s * 0.1, 0.3 + s * 0.15, 0.15 + s * 0.2];
  } else {
    // Gray to white
    const s = (t - 0.75) / 0.25;
    return [0.5 + s * 0.4, 0.45 + s * 0.45, 0.35 + s * 0.55];
  }
}

export function TerrainMesh({ heightmap }: TerrainMeshProps) {
  const geometry = useMemo(() => {
    const { width, height, data, bounds, min_elevation, max_elevation } = heightmap;

    // Downsample if too large
    const maxRes = 100;
    const stepX = Math.max(1, Math.floor(width / maxRes));
    const stepY = Math.max(1, Math.floor(height / maxRes));
    const resW = Math.floor(width / stepX);
    const resH = Math.floor(height / stepY);

    const worldW = (bounds[2] - bounds[0]) * SCALE;
    const worldH = (bounds[3] - bounds[1]) * SCALE;

    const geo = new THREE.PlaneGeometry(worldW, worldH, resW - 1, resH - 1);
    const positions = geo.attributes.position.array;
    const colors = new Float32Array(positions.length);
    const elevRange = max_elevation - min_elevation || 1;

    for (let j = 0; j < resH; j++) {
      for (let i = 0; i < resW; i++) {
        const srcIdx = (j * stepY) * width + (i * stepX);
        const dstIdx = j * resW + i;
        const elevation = data[srcIdx] || 0;

        // PlaneGeometry is X-Y, we'll rotate to be X-Z in scene
        // Set Z to elevation
        positions[dstIdx * 3 + 2] = elevation * SCALE;

        // Vertex color
        const t = (elevation - min_elevation) / elevRange;
        const [r, g, b] = elevationColor(t);
        colors[dstIdx * 3] = r;
        colors[dstIdx * 3 + 1] = g;
        colors[dstIdx * 3 + 2] = b;
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [heightmap]);

  // Position terrain at the center of its bounds
  const centerX = ((heightmap.bounds[0] + heightmap.bounds[2]) / 2) * SCALE;
  const centerY = (-(heightmap.bounds[1] + heightmap.bounds[3]) / 2) * SCALE;

  return (
    <mesh
      geometry={geometry}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[centerX, 0, centerY]}
    >
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.9}
        metalness={0.0}
      />
    </mesh>
  );
}
