/**
 * 3D Scene - The Visual Heart of the Simulation
 *
 * This uses React Three Fiber (R3F) which is React bindings for Three.js.
 *
 * Key Three.js concepts:
 * 1. Scene: The 3D world container
 * 2. Camera: Our viewpoint (we use OrbitControls to look around)
 * 3. Mesh: A visible object (geometry + material)
 * 4. Lights: Illuminate the scene
 *
 * For simulation visualization:
 * - Target: Red sphere
 * - Interceptor: Blue cone (pointing in velocity direction)
 * - Trails: Line showing path history
 * - Grid: Reference for scale/position
 */

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { EntityState, SimStateEvent } from '../types';

// Scale factor: sim uses meters, we scale down for visualization
const SCALE = 0.001; // 1 unit = 1km

interface EntityProps {
  entity: EntityState;
  trail: THREE.Vector3[];
}

/**
 * Target visualization - a red sphere
 */
function Target({ entity, trail }: EntityProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Convert sim coordinates to Three.js coordinates
  const position: [number, number, number] = [
    entity.position.x * SCALE,
    entity.position.z * SCALE, // Z (up) becomes Y in Three.js
    -entity.position.y * SCALE, // Y (north) becomes -Z
  ];

  return (
    <group>
      {/* Target sphere */}
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#ef4444" emissive="#991b1b" emissiveIntensity={0.3} />
      </mesh>

      {/* Label */}
      <Text
        position={[position[0], position[1] + 0.3, position[2]]}
        fontSize={0.15}
        color="#ef4444"
        anchorX="center"
      >
        TARGET
      </Text>

      {/* Trail */}
      {trail.length > 1 && (
        <Line points={trail} color="#ef4444" lineWidth={2} opacity={0.5} transparent />
      )}
    </group>
  );
}

/**
 * Interceptor visualization - a blue cone pointing in velocity direction
 */
function Interceptor({ entity, trail }: EntityProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const position: [number, number, number] = [
    entity.position.x * SCALE,
    entity.position.z * SCALE,
    -entity.position.y * SCALE,
  ];

  // Calculate rotation to point cone in velocity direction
  useFrame(() => {
    if (!meshRef.current) return;

    const vel = entity.velocity;
    if (vel.x === 0 && vel.y === 0 && vel.z === 0) return;

    // Create direction vector in Three.js space
    const dir = new THREE.Vector3(vel.x, vel.z, -vel.y).normalize();

    // Point the cone in the velocity direction
    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4();

    matrix.lookAt(new THREE.Vector3(), dir, up);
    quaternion.setFromRotationMatrix(matrix);

    // Cone points up by default, rotate to point forward
    const correction = new THREE.Quaternion();
    correction.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
    quaternion.multiply(correction);

    meshRef.current.quaternion.copy(quaternion);
  });

  return (
    <group>
      {/* Interceptor cone */}
      <mesh ref={meshRef} position={position}>
        <coneGeometry args={[0.1, 0.3, 8]} />
        <meshStandardMaterial color="#3b82f6" emissive="#1d4ed8" emissiveIntensity={0.3} />
      </mesh>

      {/* Label */}
      <Text
        position={[position[0], position[1] + 0.3, position[2]]}
        fontSize={0.15}
        color="#3b82f6"
        anchorX="center"
      >
        INTERCEPTOR
      </Text>

      {/* Trail */}
      {trail.length > 1 && (
        <Line points={trail} color="#3b82f6" lineWidth={2} opacity={0.5} transparent />
      )}
    </group>
  );
}

interface SceneContentProps {
  state: SimStateEvent | null;
  trails: Map<string, THREE.Vector3[]>;
}

function SceneContent({ state, trails }: SceneContentProps) {
  const target = state?.entities.find((e) => e.type === 'target');
  const interceptor = state?.entities.find((e) => e.type === 'interceptor');

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.3} />

      {/* Camera controls - lets user rotate/zoom */}
      <OrbitControls
        makeDefault
        minDistance={1}
        maxDistance={50}
        target={[2.5, 0.5, 0]} // Center on action area
      />

      {/* Ground grid - 1 unit = 1km */}
      <Grid
        args={[20, 20]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#374151"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#6b7280"
        fadeDistance={30}
        position={[0, 0, 0]}
      />

      {/* Axis helper for orientation */}
      <axesHelper args={[2]} />

      {/* Entities */}
      {target && <Target entity={target} trail={trails.get(target.id) || []} />}
      {interceptor && <Interceptor entity={interceptor} trail={trails.get(interceptor.id) || []} />}
    </>
  );
}

interface SimulationSceneProps {
  state: SimStateEvent | null;
}

export function SimulationScene({ state }: SimulationSceneProps) {
  // Maintain trail history
  const trailsRef = useRef<Map<string, THREE.Vector3[]>>(new Map());

  // Update trails when state changes
  useMemo(() => {
    if (!state) {
      trailsRef.current.clear();
      return;
    }

    for (const entity of state.entities) {
      const pos = new THREE.Vector3(
        entity.position.x * SCALE,
        entity.position.z * SCALE,
        -entity.position.y * SCALE
      );

      let trail = trailsRef.current.get(entity.id);
      if (!trail) {
        trail = [];
        trailsRef.current.set(entity.id, trail);
      }

      // Add point if moved enough (avoid cluttering with tiny movements)
      const lastPoint = trail[trail.length - 1];
      if (!lastPoint || pos.distanceTo(lastPoint) > 0.05) {
        trail.push(pos);

        // Limit trail length
        if (trail.length > 500) {
          trail.shift();
        }
      }
    }
  }, [state?.tick]);

  return (
    <Canvas
      camera={{
        position: [8, 4, 8],
        fov: 50,
        near: 0.1,
        far: 1000,
      }}
      style={{ background: '#111827' }}
    >
      <SceneContent state={state} trails={trailsRef.current} />
    </Canvas>
  );
}
