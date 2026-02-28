/**
 * Explosion - Expanding particle flash on intercept
 *
 * Triggered when an intercept occurs. Expands and fades over ~0.6 seconds.
 * Uses additive blending for a bright flash effect.
 */

import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface ExplosionProps {
  position: [number, number, number];
  onComplete?: () => void;
}

export function Explosion({ position, onComplete }: ExplosionProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [scale, setScale] = useState(0.1);
  const [opacity, setOpacity] = useState(1.0);
  const [done, setDone] = useState(false);

  useFrame((_, delta) => {
    if (done) return;

    const newScale = scale + delta * 4;
    const newOpacity = Math.max(0, opacity - delta * 2.5);

    setScale(newScale);
    setOpacity(newOpacity);

    if (meshRef.current) {
      meshRef.current.scale.setScalar(newScale);
    }

    if (newOpacity <= 0) {
      setDone(true);
      onComplete?.();
    }
  });

  if (done) return null;

  return (
    <group position={position}>
      {/* Core flash */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.2, 12, 12]} />
        <meshBasicMaterial
          color="#ff6600"
          transparent
          opacity={opacity}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Outer glow ring */}
      <mesh scale={scale * 1.5}>
        <sphereGeometry args={[0.15, 8, 8]} />
        <meshBasicMaterial
          color="#ffaa00"
          transparent
          opacity={opacity * 0.4}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* White flash center */}
      <mesh scale={scale * 0.5}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={opacity * 0.8}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
