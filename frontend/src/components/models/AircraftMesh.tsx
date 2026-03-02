/**
 * AircraftMesh — Target/threat aircraft geometry
 *
 * Recognizable aircraft silhouette: pointed nose → fuselage → swept wings →
 * horizontal stabilizer → vertical tail(s). Oriented along +Y.
 *
 * Can represent different threat types through color and scale.
 */

import { useMemo } from 'react';
import * as THREE from 'three';

interface AircraftMeshProps {
  color: string;
  emissive: string;
  emissiveIntensity?: number;
  opacity?: number;
}

export function AircraftMesh({ color, emissive, emissiveIntensity = 0.3, opacity = 1 }: AircraftMeshProps) {
  const transparent = opacity < 1;

  // Fuselage — smooth lathe profile
  const fuselageGeometry = useMemo(() => {
    const points = [
      new THREE.Vector2(0, 0.22),      // nose tip
      new THREE.Vector2(0.012, 0.19),  // nose shoulder
      new THREE.Vector2(0.03, 0.14),   // canopy area
      new THREE.Vector2(0.038, 0.06),  // widest
      new THREE.Vector2(0.036, -0.04), // mid
      new THREE.Vector2(0.032, -0.10), // rear
      new THREE.Vector2(0.025, -0.14), // tail taper
      new THREE.Vector2(0.018, -0.16), // tail
    ];
    return new THREE.LatheGeometry(points, 10);
  }, []);

  // Main wings — swept delta shape
  const wingGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    // Right wing (mirrored for left)
    shape.moveTo(0, 0);          // root leading edge
    shape.lineTo(0.22, -0.03);   // tip leading edge (swept back)
    shape.lineTo(0.18, -0.08);   // tip trailing edge
    shape.lineTo(0, -0.06);      // root trailing edge
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.005,
      bevelEnabled: false,
    });
  }, []);

  // Horizontal stabilizer
  const stabGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(0.08, -0.01);
    shape.lineTo(0.06, -0.035);
    shape.lineTo(0, -0.025);
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.003,
      bevelEnabled: false,
    });
  }, []);

  // Vertical tail fin
  const tailGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(0, 0.08);
    shape.lineTo(-0.04, 0.06);
    shape.lineTo(-0.03, 0);
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, {
      depth: 0.004,
      bevelEnabled: false,
    });
  }, []);

  const matProps = {
    color,
    emissive,
    emissiveIntensity,
    roughness: 0.35,
    metalness: 0.5,
    transparent,
    opacity,
  };

  return (
    <group>
      {/* Fuselage */}
      <mesh geometry={fuselageGeometry}>
        <meshStandardMaterial {...matProps} />
      </mesh>

      {/* Canopy — small glass dome */}
      <mesh position={[0, 0.12, 0.03]} rotation={[0.3, 0, 0]}>
        <sphereGeometry args={[0.022, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial
          color="#88ccff"
          emissive="#3b82f6"
          emissiveIntensity={0.4}
          roughness={0.1}
          metalness={0.8}
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* Right wing */}
      <mesh
        position={[0.035, 0.02, -0.002]}
        geometry={wingGeometry}
      >
        <meshStandardMaterial {...matProps} side={THREE.DoubleSide} emissiveIntensity={emissiveIntensity * 0.6} />
      </mesh>

      {/* Left wing (mirrored) */}
      <mesh
        position={[-0.035, 0.02, 0.002]}
        rotation={[0, Math.PI, 0]}
        geometry={wingGeometry}
      >
        <meshStandardMaterial {...matProps} side={THREE.DoubleSide} emissiveIntensity={emissiveIntensity * 0.6} />
      </mesh>

      {/* Right stabilizer */}
      <mesh
        position={[0.025, -0.13, -0.001]}
        geometry={stabGeometry}
      >
        <meshStandardMaterial {...matProps} side={THREE.DoubleSide} emissiveIntensity={emissiveIntensity * 0.4} />
      </mesh>

      {/* Left stabilizer (mirrored) */}
      <mesh
        position={[-0.025, -0.13, 0.001]}
        rotation={[0, Math.PI, 0]}
        geometry={stabGeometry}
      >
        <meshStandardMaterial {...matProps} side={THREE.DoubleSide} emissiveIntensity={emissiveIntensity * 0.4} />
      </mesh>

      {/* Vertical tail */}
      <mesh
        position={[0, -0.12, -0.002]}
        geometry={tailGeometry}
      >
        <meshStandardMaterial {...matProps} side={THREE.DoubleSide} emissiveIntensity={emissiveIntensity * 0.5} />
      </mesh>

      {/* Engine exhaust glow (twin) */}
      <mesh position={[0.015, -0.155, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.012, 6]} />
        <meshBasicMaterial
          color="#ff4400"
          transparent
          opacity={transparent ? opacity * 0.5 : 0.6}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh position={[-0.015, -0.155, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.012, 6]} />
        <meshBasicMaterial
          color="#ff4400"
          transparent
          opacity={transparent ? opacity * 0.5 : 0.6}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}
