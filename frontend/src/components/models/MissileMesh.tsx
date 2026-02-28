/**
 * MissileMesh - Low-poly missile geometry for interceptors
 *
 * Composed of: nose cone + body cylinder + 4 fins + exhaust nozzle
 * Oriented along +Y axis (local), rotated by parent to match velocity.
 */

import { useMemo } from 'react';
import * as THREE from 'three';

interface MissileMeshProps {
  color: string;
  emissive: string;
  emissiveIntensity?: number;
}

export function MissileMesh({ color, emissive, emissiveIntensity = 0.3 }: MissileMeshProps) {
  const finGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(0.08, 0);
    shape.lineTo(0.03, 0.06);
    shape.lineTo(0, 0.06);
    shape.closePath();
    const geo = new THREE.ShapeGeometry(shape);
    return geo;
  }, []);

  return (
    <group>
      {/* Nose cone */}
      <mesh position={[0, 0.18, 0]}>
        <coneGeometry args={[0.035, 0.12, 6]} />
        <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={emissiveIntensity} />
      </mesh>

      {/* Body */}
      <mesh position={[0, 0.05, 0]}>
        <cylinderGeometry args={[0.035, 0.038, 0.2, 6]} />
        <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={emissiveIntensity * 0.8} />
      </mesh>

      {/* Fins (4x at 90 degree intervals) */}
      {[0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((angle, i) => (
        <mesh
          key={i}
          position={[
            Math.sin(angle) * 0.035,
            -0.04,
            Math.cos(angle) * 0.035,
          ]}
          rotation={[0, -angle, 0]}
          geometry={finGeometry}
        >
          <meshStandardMaterial
            color={color}
            emissive={emissive}
            emissiveIntensity={emissiveIntensity * 0.6}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {/* Exhaust nozzle */}
      <mesh position={[0, -0.06, 0]}>
        <cylinderGeometry args={[0.025, 0.032, 0.03, 6]} />
        <meshStandardMaterial color="#44403c" emissive="#78716c" emissiveIntensity={0.2} />
      </mesh>
    </group>
  );
}
