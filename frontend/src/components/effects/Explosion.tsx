/**
 * Explosion — Multi-layer expanding blast on intercept
 *
 * Three-phase flash: initial white core → orange fireball → fading debris ring.
 * Uses additive blending for bright, camera-facing glow.
 */

import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface ExplosionProps {
  position: [number, number, number];
  onComplete?: () => void;
}

const DURATION = 0.8;

export function Explosion({ position, onComplete }: ExplosionProps) {
  const coreRef = useRef<THREE.Mesh>(null);
  const outerRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const ageRef = useRef(0);
  const [done, setDone] = useState(false);

  useFrame((_, delta) => {
    if (done) return;
    ageRef.current += delta;
    const t = ageRef.current / DURATION;

    if (t >= 1) {
      setDone(true);
      onComplete?.();
      return;
    }

    // Core flash: fast expand then shrink
    if (coreRef.current) {
      const coreScale = t < 0.15
        ? THREE.MathUtils.lerp(0.1, 1.2, t / 0.15)
        : THREE.MathUtils.lerp(1.2, 0.3, (t - 0.15) / 0.85);
      coreRef.current.scale.setScalar(coreScale);
      (coreRef.current.material as THREE.MeshBasicMaterial).opacity = t < 0.2 ? 1 : Math.max(0, 1 - (t - 0.2) / 0.5);
    }

    // Outer fireball: expands slower, fades later
    if (outerRef.current) {
      const outerScale = THREE.MathUtils.lerp(0.2, 2.0, Math.min(t * 2, 1));
      outerRef.current.scale.setScalar(outerScale);
      (outerRef.current.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.5 - t * 0.6);
    }

    // Shockwave ring: expands outward
    if (ringRef.current) {
      const ringScale = THREE.MathUtils.lerp(0.5, 3.0, t);
      ringRef.current.scale.setScalar(ringScale);
      (ringRef.current.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.4 * (1 - t));
    }
  });

  if (done) return null;

  return (
    <group position={position}>
      {/* White-hot core */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[0.15, 12, 12]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={1}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Orange fireball */}
      <mesh ref={outerRef}>
        <sphereGeometry args={[0.12, 10, 10]} />
        <meshBasicMaterial
          color="#ff6600"
          transparent
          opacity={0.5}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Expanding shockwave ring */}
      <mesh ref={ringRef} rotation={[Math.random() * Math.PI, Math.random() * Math.PI, 0]}>
        <torusGeometry args={[0.1, 0.02, 6, 24]} />
        <meshBasicMaterial
          color="#ffaa00"
          transparent
          opacity={0.4}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
