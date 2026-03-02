/**
 * Atmosphere — Subtle blue edge glow around the Earth globe.
 * Uses a slightly larger BackSide sphere with low opacity for a soft halo.
 */

import * as THREE from 'three';

interface AtmosphereProps {
  radius?: number;
}

export function Atmosphere({ radius = 5 }: AtmosphereProps) {
  return (
    <mesh>
      <sphereGeometry args={[radius * 1.02, 64, 64]} />
      <meshBasicMaterial
        color="#3b82f6"
        transparent
        opacity={0.06}
        side={THREE.BackSide}
        depthWrite={false}
      />
    </mesh>
  );
}
