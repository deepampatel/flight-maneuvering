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
 * Phase 3 enhancements:
 * - Multiple interceptors with distinct colors
 * - Gradient trails with time markers
 * - Enhanced trail visualization
 */

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { EntityState, SimStateEvent, InterceptGeometry, Vec3 } from '../types';

// Scale factor: sim uses meters, we scale down for visualization
const SCALE = 0.001; // 1 unit = 1km

// Color palette for multiple interceptors
const INTERCEPTOR_COLORS = [
  '#3b82f6', // Blue
  '#22c55e', // Green
  '#06b6d4', // Cyan
  '#a855f7', // Purple
  '#f97316', // Orange
  '#eab308', // Yellow
  '#ec4899', // Pink
  '#14b8a6', // Teal
];

const INTERCEPTOR_EMISSIVE = [
  '#1d4ed8',
  '#15803d',
  '#0891b2',
  '#7c3aed',
  '#c2410c',
  '#a16207',
  '#be185d',
  '#0f766e',
];

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

interface InterceptorProps extends EntityProps {
  colorIndex?: number;
}

/**
 * Interceptor visualization - a cone pointing in velocity direction
 * Supports multiple interceptors with distinct colors
 */
function Interceptor({ entity, trail, colorIndex = 0 }: InterceptorProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Get color for this interceptor
  const color = INTERCEPTOR_COLORS[colorIndex % INTERCEPTOR_COLORS.length];
  const emissive = INTERCEPTOR_EMISSIVE[colorIndex % INTERCEPTOR_EMISSIVE.length];

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
        <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={0.3} />
      </mesh>

      {/* Label */}
      <Text
        position={[position[0], position[1] + 0.3, position[2]]}
        fontSize={0.12}
        color={color}
        anchorX="center"
      >
        {entity.id}
      </Text>

      {/* Trail with gradient effect */}
      {trail.length > 1 && (
        <GradientTrail points={trail} color={color} />
      )}
    </group>
  );
}

/**
 * Gradient trail component - fades from start to end
 */
function GradientTrail({ points, color }: { points: THREE.Vector3[]; color: string }) {
  // Create time markers every ~50 points (about 1 second at 50Hz)
  const markerIndices = useMemo(() => {
    const indices: number[] = [];
    for (let i = 50; i < points.length; i += 50) {
      indices.push(i);
    }
    return indices;
  }, [points.length]);

  return (
    <group>
      {/* Main trail line */}
      <Line
        points={points}
        color={color}
        lineWidth={2}
        opacity={0.6}
        transparent
      />

      {/* Time markers (small spheres along trail) */}
      {markerIndices.map((idx) => (
        <mesh key={idx} position={points[idx]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshBasicMaterial color={color} opacity={0.4} transparent />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Intercept Point Marker - Shows predicted collision location
 */
function InterceptPointMarker({ point, collision }: { point: Vec3; collision: boolean }) {
  const position: [number, number, number] = [
    point.x * SCALE,
    point.z * SCALE,
    -point.y * SCALE,
  ];

  return (
    <group position={position}>
      {/* Wireframe sphere at intercept point */}
      <mesh>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshBasicMaterial
          color={collision ? '#22c55e' : '#f97316'}
          wireframe
          transparent
          opacity={0.8}
        />
      </mesh>
      {/* Inner glow */}
      <mesh>
        <sphereGeometry args={[0.06, 12, 12]} />
        <meshBasicMaterial
          color={collision ? '#22c55e' : '#f97316'}
          transparent
          opacity={0.4}
        />
      </mesh>
      <Text
        position={[0, 0.25, 0]}
        fontSize={0.1}
        color={collision ? '#22c55e' : '#f97316'}
        anchorX="center"
      >
        INTERCEPT
      </Text>
    </group>
  );
}

/**
 * Lead Pursuit Line - Shows optimal heading to intercept
 */
function LeadPursuitLine({
  from,
  to,
  collision,
}: {
  from: Vec3;
  to: Vec3;
  collision: boolean;
}) {
  const points = useMemo(() => {
    return [
      new THREE.Vector3(from.x * SCALE, from.z * SCALE, -from.y * SCALE),
      new THREE.Vector3(to.x * SCALE, to.z * SCALE, -to.y * SCALE),
    ];
  }, [from, to]);

  return (
    <Line
      points={points}
      color={collision ? '#22c55e' : '#f97316'}
      lineWidth={1.5}
      dashed
      dashSize={0.1}
      gapSize={0.05}
      opacity={0.6}
      transparent
    />
  );
}

interface SceneContentProps {
  state: SimStateEvent | null;
  trails: Map<string, THREE.Vector3[]>;
  interceptGeometry?: InterceptGeometry[] | null;
}

function SceneContent({ state, trails, interceptGeometry }: SceneContentProps) {
  const target = state?.entities.find((e) => e.type === 'target');
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];

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

      {/* Target */}
      {target && <Target entity={target} trail={trails.get(target.id) || []} />}

      {/* All Interceptors */}
      {interceptors.map((interceptor, idx) => (
        <Interceptor
          key={interceptor.id}
          entity={interceptor}
          trail={trails.get(interceptor.id) || []}
          colorIndex={idx}
        />
      ))}

      {/* Intercept Geometry Visualization */}
      {interceptGeometry && interceptGeometry.map((geom) => {
        const interceptor = interceptors.find((i) => i.id === geom.interceptor_id);
        if (!interceptor || !geom.intercept_point) return null;

        return (
          <group key={`geom-${geom.interceptor_id}`}>
            {/* Intercept point marker */}
            <InterceptPointMarker
              point={geom.intercept_point}
              collision={geom.collision_course}
            />
            {/* Lead pursuit line from interceptor to intercept point */}
            <LeadPursuitLine
              from={interceptor.position}
              to={geom.intercept_point}
              collision={geom.collision_course}
            />
          </group>
        );
      })}
    </>
  );
}

interface SimulationSceneProps {
  state: SimStateEvent | null;
  interceptGeometry?: InterceptGeometry[] | null;
}

export function SimulationScene({ state, interceptGeometry }: SimulationSceneProps) {
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
      <SceneContent state={state} trails={trailsRef.current} interceptGeometry={interceptGeometry} />
    </Canvas>
  );
}
