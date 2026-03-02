/**
 * ExhaustTrail - Particle-based exhaust trail behind missiles
 *
 * Uses a Points buffer that spawns particles at the entity tail
 * and fades them over time. Additive blending for glow effect.
 */

import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const MAX_PARTICLES = 30;
const PARTICLE_LIFETIME = 1.0; // seconds
const SPAWN_RATE = 25; // particles per second

interface ExhaustTrailProps {
  position: [number, number, number];
  velocity: [number, number, number];
  color?: string;
  active?: boolean;
}

export function ExhaustTrail({ position, velocity, color = '#ff8800', active = true }: ExhaustTrailProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const spawnTimerRef = useRef(0);

  // Particle data stored in typed arrays
  const { positions, ages } = useMemo(() => {
    const positions = new Float32Array(MAX_PARTICLES * 3);
    const ages = new Float32Array(MAX_PARTICLES).fill(PARTICLE_LIFETIME + 1); // Start expired
    return { positions, ages };
  }, []);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);

  // Cleanup
  useEffect(() => {
    return () => { geometry.dispose(); };
  }, [geometry]);

  const frameCountRef = useRef(0);

  useFrame((_, delta) => {
    if (!pointsRef.current) return;

    // Skip every other frame for performance
    frameCountRef.current++;
    if (frameCountRef.current % 2 !== 0) return;
    delta *= 2; // compensate for skipped frame

    // Compute tail position (behind entity in velocity direction)
    const speed = Math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2);
    const tailOffset = speed > 0.001 ? 0.15 : 0;
    const tailX = position[0] - (velocity[0] / (speed || 1)) * tailOffset;
    const tailY = position[1] - (velocity[1] / (speed || 1)) * tailOffset;
    const tailZ = position[2] - (velocity[2] / (speed || 1)) * tailOffset;

    // Spawn new particles
    if (active) {
      spawnTimerRef.current += delta;
      const spawnInterval = 1.0 / SPAWN_RATE;

      while (spawnTimerRef.current >= spawnInterval) {
        spawnTimerRef.current -= spawnInterval;

        // Find an expired particle to reuse
        for (let i = 0; i < MAX_PARTICLES; i++) {
          if (ages[i] > PARTICLE_LIFETIME) {
            const idx = i * 3;
            positions[idx] = tailX + (Math.random() - 0.5) * 0.02;
            positions[idx + 1] = tailY + (Math.random() - 0.5) * 0.02;
            positions[idx + 2] = tailZ + (Math.random() - 0.5) * 0.02;
            ages[i] = 0;
            break;
          }
        }
      }
    }

    // Update all particles
    for (let i = 0; i < MAX_PARTICLES; i++) {
      ages[i] += delta;
      const t = ages[i] / PARTICLE_LIFETIME;
      if (t <= 1) {
        // Slight upward drift
        positions[i * 3 + 1] += delta * 0.05;
        // Slight random drift
        positions[i * 3] += (Math.random() - 0.5) * delta * 0.1;
        positions[i * 3 + 2] += (Math.random() - 0.5) * delta * 0.1;
      }
    }

    // Update the buffer
    const attr = geometry.getAttribute('position') as THREE.BufferAttribute;
    attr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={0.06}
        color={color}
        transparent
        opacity={0.5}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
}
