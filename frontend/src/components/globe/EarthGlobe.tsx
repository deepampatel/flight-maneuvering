/**
 * EarthGlobe — Dark-styled 3D Earth with NASA night lights texture.
 * Built with plain Three.js primitives — no extra dependencies.
 */

import { useTexture } from '@react-three/drei';
import * as THREE from 'three';

interface EarthGlobeProps {
  radius?: number;
}

export function EarthGlobe({ radius = 5 }: EarthGlobeProps) {
  const nightMap = useTexture('/textures/earth_night.jpg');

  return (
    <mesh>
      <sphereGeometry args={[radius, 64, 64]} />
      <meshStandardMaterial
        map={nightMap}
        emissiveMap={nightMap}
        emissive="#ffffff"
        emissiveIntensity={1.0}
        roughness={1}
        metalness={0}
        side={THREE.FrontSide}
      />
    </mesh>
  );
}
