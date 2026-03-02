/**
 * MissileMesh — Interceptor missile geometry
 *
 * Sleek aerodynamic shape: tapered nose → smooth body → 4 cruciform fins → nozzle
 * Uses LatheGeometry for a smooth body profile and proper fin shapes.
 * Oriented along +Y axis, parent rotates to match velocity.
 */

import { useMemo } from 'react';
import * as THREE from 'three';

interface MissileMeshProps {
  color: string;
  emissive: string;
  emissiveIntensity?: number;
}

export function MissileMesh({ color, emissive, emissiveIntensity = 0.3 }: MissileMeshProps) {
  // Smooth body profile via lathe
  const bodyGeometry = useMemo(() => {
    const points = [
      new THREE.Vector2(0, 0.28),       // nose tip
      new THREE.Vector2(0.015, 0.25),   // nose shoulder
      new THREE.Vector2(0.032, 0.20),   // nose base
      new THREE.Vector2(0.038, 0.12),   // mid body
      new THREE.Vector2(0.038, -0.02),  // rear body
      new THREE.Vector2(0.035, -0.06),  // taper start
      new THREE.Vector2(0.028, -0.08),  // nozzle
      new THREE.Vector2(0.020, -0.09),  // nozzle exit
    ];
    return new THREE.LatheGeometry(points, 12);
  }, []);

  // Cruciform fins — 4 swept delta shapes
  const finGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(0.09, -0.01);
    shape.lineTo(0.06, 0.07);
    shape.lineTo(0.02, 0.08);
    shape.lineTo(0, 0.06);
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.003,
      bevelEnabled: false,
    });
  }, []);

  // Small canard fins near nose
  const canardGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(0.04, 0);
    shape.lineTo(0.02, 0.03);
    shape.lineTo(0, 0.03);
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.002,
      bevelEnabled: false,
    });
  }, []);

  return (
    <group>
      {/* Main body — lathe profile */}
      <mesh geometry={bodyGeometry}>
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          roughness={0.3}
          metalness={0.6}
        />
      </mesh>

      {/* Seeker dome — glowing tip */}
      <mesh position={[0, 0.28, 0]}>
        <sphereGeometry args={[0.016, 8, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.9} />
      </mesh>

      {/* Rear fins — 4 at 90° intervals */}
      {[0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((angle, i) => (
        <mesh
          key={`fin-${i}`}
          position={[
            Math.sin(angle) * 0.036,
            -0.06,
            Math.cos(angle) * 0.036,
          ]}
          rotation={[0, -angle, 0]}
          geometry={finGeometry}
        >
          <meshStandardMaterial
            color={color}
            emissive={emissive}
            emissiveIntensity={emissiveIntensity * 0.5}
            roughness={0.4}
            metalness={0.5}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* Canards near nose — 4 smaller fins */}
      {[Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4].map((angle, i) => (
        <mesh
          key={`canard-${i}`}
          position={[
            Math.sin(angle) * 0.032,
            0.15,
            Math.cos(angle) * 0.032,
          ]}
          rotation={[0, -angle, 0]}
          geometry={canardGeometry}
        >
          <meshStandardMaterial
            color={color}
            emissive={emissive}
            emissiveIntensity={emissiveIntensity * 0.3}
            roughness={0.5}
            metalness={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* Nozzle interior glow */}
      <mesh position={[0, -0.088, 0]}>
        <circleGeometry args={[0.018, 8]} />
        <meshBasicMaterial
          color="#ff6600"
          transparent
          opacity={0.7}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Engine glow point light */}
      <pointLight
        position={[0, -0.1, 0]}
        color={color}
        intensity={0.3}
        distance={0.8}
      />
    </group>
  );
}
