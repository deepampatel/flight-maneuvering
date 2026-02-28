/**
 * AircraftMesh - Low-poly aircraft geometry for targets
 *
 * Composed of: fuselage + 2 wings + horizontal stabilizer + vertical tail
 * Oriented along +Y axis (local), rotated by parent to match velocity.
 */

import * as THREE from 'three';

interface AircraftMeshProps {
  color: string;
  emissive: string;
  emissiveIntensity?: number;
  opacity?: number;
}

export function AircraftMesh({ color, emissive, emissiveIntensity = 0.3, opacity = 1 }: AircraftMeshProps) {
  const transparent = opacity < 1;

  return (
    <group>
      {/* Fuselage */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.03, 0.04, 0.28, 6]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          transparent={transparent}
          opacity={opacity}
        />
      </mesh>

      {/* Nose cone */}
      <mesh position={[0, 0.16, 0]}>
        <coneGeometry args={[0.03, 0.08, 6]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          transparent={transparent}
          opacity={opacity}
        />
      </mesh>

      {/* Wings (swept delta) */}
      <mesh position={[0, 0.02, 0]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.3, 0.01, 0.1]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity * 0.7}
          transparent={transparent}
          opacity={opacity}
        />
      </mesh>

      {/* Horizontal stabilizer */}
      <mesh position={[0, -0.11, 0]}>
        <boxGeometry args={[0.12, 0.008, 0.05]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity * 0.6}
          transparent={transparent}
          opacity={opacity}
        />
      </mesh>

      {/* Vertical tail */}
      <mesh position={[0, -0.08, 0]} rotation={[0, 0, Math.PI / 2]}>
        <boxGeometry args={[0.08, 0.008, 0.05]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity * 0.6}
          transparent={transparent}
          opacity={opacity}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}
