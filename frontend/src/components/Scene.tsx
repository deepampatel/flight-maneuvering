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

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { EntityState, SimStateEvent, InterceptGeometry, Vec3, AssignmentResult } from '../types';

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

// Phase 5: Color palette for multiple targets (shades of red/orange)
const TARGET_COLORS = [
  '#ef4444', // Red
  '#f97316', // Orange
  '#dc2626', // Dark red
  '#ea580c', // Dark orange
];

const TARGET_EMISSIVE = [
  '#991b1b',
  '#c2410c',
  '#7f1d1d',
  '#9a3412',
];

interface EntityProps {
  entity: EntityState;
  trail: THREE.Vector3[];
}

interface TargetProps extends EntityProps {
  colorIndex?: number;
  isIntercepted?: boolean;
}

/**
 * Target visualization - a sphere with color based on index
 * Phase 5: Supports multiple targets with distinct colors
 */
function Target({ entity, trail, colorIndex = 0, isIntercepted = false }: TargetProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Get color for this target
  const color = TARGET_COLORS[colorIndex % TARGET_COLORS.length];
  const emissive = TARGET_EMISSIVE[colorIndex % TARGET_EMISSIVE.length];

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
        <meshStandardMaterial
          color={isIntercepted ? '#6b7280' : color}
          emissive={isIntercepted ? '#374151' : emissive}
          emissiveIntensity={isIntercepted ? 0.1 : 0.3}
          opacity={isIntercepted ? 0.5 : 1}
          transparent={isIntercepted}
        />
      </mesh>

      {/* Label */}
      <Text
        position={[position[0], position[1] + 0.3, position[2]]}
        fontSize={0.12}
        color={isIntercepted ? '#6b7280' : color}
        anchorX="center"
      >
        {entity.id}
      </Text>

      {/* Trail - always visible, brighter when intercepted for visibility */}
      {trail.length > 1 && (
        <Line
          points={trail}
          color={isIntercepted ? '#9ca3af' : color}
          lineWidth={3}
          opacity={0.8}
          transparent
        />
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
      {/* Main trail line - brighter for visibility */}
      <Line
        points={points}
        color={color}
        lineWidth={3}
        opacity={0.8}
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

/**
 * Standalone trail component - renders trails even when entities are gone
 * This ensures flight paths remain visible after simulation ends
 */
function PersistentTrail({
  points,
  isTarget,
  colorIndex,
}: {
  points: THREE.Vector3[];
  isTarget: boolean;
  colorIndex: number;
}) {
  if (points.length < 2) return null;

  const color = isTarget
    ? TARGET_COLORS[colorIndex % TARGET_COLORS.length]
    : INTERCEPTOR_COLORS[colorIndex % INTERCEPTOR_COLORS.length];

  return (
    <group>
      <Line
        points={points}
        color={color}
        lineWidth={2}
        opacity={0.7}
        transparent
      />
      {/* End point marker */}
      <mesh position={points[points.length - 1]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshBasicMaterial color={color} opacity={0.8} transparent />
      </mesh>
      {/* Start point marker */}
      <mesh position={points[0]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color={color} opacity={0.5} transparent />
      </mesh>
    </group>
  );
}

interface SceneContentProps {
  state: SimStateEvent | null;
  trails: Map<string, THREE.Vector3[]>;
  interceptGeometry?: InterceptGeometry[] | null;
  assignments?: AssignmentResult | null;
}

function SceneContent({ state, trails, interceptGeometry, assignments }: SceneContentProps) {
  // Phase 5: Support multiple targets
  const targets = state?.entities.filter((e) => e.type === 'target') || [];
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];

  // Get all entity IDs currently in state
  const currentEntityIds = new Set(state?.entities.map((e) => e.id) || []);

  // Track which targets have been intercepted
  const interceptedTargetIds = new Set(
    state?.intercepted_pairs?.map((pair) => pair[1]) || []
  );

  // Track which interceptors have hit their target
  const interceptedInterceptorIds = new Set(
    state?.intercepted_pairs?.map((pair) => pair[0]) || []
  );

  // Build a map of interceptor -> assigned target for filtering geometry
  const assignmentMap = useMemo(() => {
    const map = new Map<string, string>();
    if (assignments?.assignments) {
      for (const a of assignments.assignments) {
        map.set(a.interceptor_id, a.target_id);
      }
    }
    return map;
  }, [assignments]);

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

      {/* All Targets - Phase 5 multi-target support */}
      {targets.map((target, idx) => (
        <Target
          key={target.id}
          entity={target}
          trail={trails.get(target.id) || []}
          colorIndex={idx}
          isIntercepted={interceptedTargetIds.has(target.id)}
        />
      ))}

      {/* All Interceptors */}
      {interceptors.map((interceptor, idx) => (
        <Interceptor
          key={interceptor.id}
          entity={interceptor}
          trail={trails.get(interceptor.id) || []}
          colorIndex={idx}
        />
      ))}

      {/* Persistent trails - render trails for entities not currently in state
          This ensures flight paths remain visible after simulation ends */}
      {Array.from(trails.entries()).map(([entityId, points]) => {
        // Skip if entity is currently being rendered (its component handles the trail)
        if (currentEntityIds.has(entityId)) return null;

        // Determine if this was a target or interceptor based on ID prefix
        const isTarget = entityId.startsWith('T');
        const colorIndex = parseInt(entityId.replace(/\D/g, ''), 10) - 1 || 0;

        return (
          <PersistentTrail
            key={`trail-${entityId}`}
            points={points}
            isTarget={isTarget}
            colorIndex={colorIndex}
          />
        );
      })}

      {/* Intercept Geometry Visualization - Only show for assigned targets */}
      {interceptGeometry && interceptGeometry.map((geom) => {
        const interceptor = interceptors.find((i) => i.id === geom.interceptor_id);
        if (!interceptor || !geom.intercept_point) return null;

        // Skip interceptors that have already hit their target
        if (interceptedInterceptorIds.has(geom.interceptor_id)) return null;

        // Skip intercepted targets
        if (interceptedTargetIds.has(geom.target_id)) return null;

        // Phase 5: Only show geometry for assigned target (not all targets)
        const assignedTargetId = assignmentMap.get(geom.interceptor_id);
        if (assignedTargetId && geom.target_id !== assignedTargetId) return null;

        return (
          <group key={`geom-${geom.interceptor_id}-${geom.target_id}`}>
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
  assignments?: AssignmentResult | null;
}

export function SimulationScene({ state, interceptGeometry, assignments }: SimulationSceneProps) {
  // Maintain trail history with state to trigger re-renders
  const [trails, setTrails] = useState<Map<string, THREE.Vector3[]>>(new Map());
  // Track current run_id to know when to clear trails
  const currentRunIdRef = useRef<string | null>(null);

  // Update trails when state changes
  useEffect(() => {
    if (!state) {
      // Don't clear trails when state is null - preserve them for viewing
      return;
    }

    setTrails(prevTrails => {
      // Clear trails only when a NEW run starts (different run_id)
      if (state.run_id !== currentRunIdRef.current) {
        currentRunIdRef.current = state.run_id;
        // Start fresh with empty Map for new run
        const freshTrails = new Map<string, THREE.Vector3[]>();
        for (const entity of state.entities) {
          const pos = new THREE.Vector3(
            entity.position.x * SCALE,
            entity.position.z * SCALE,
            -entity.position.y * SCALE
          );
          freshTrails.set(entity.id, [pos]);
        }
        return freshTrails;
      }

      // Create a new Map with NEW arrays (immutable update for React)
      const newTrails = new Map<string, THREE.Vector3[]>();

      for (const entity of state.entities) {
        const pos = new THREE.Vector3(
          entity.position.x * SCALE,
          entity.position.z * SCALE,
          -entity.position.y * SCALE
        );

        const existingTrail = prevTrails.get(entity.id) || [];
        const lastPoint = existingTrail[existingTrail.length - 1];

        // Add point if moved enough (avoid cluttering with tiny movements)
        if (!lastPoint || pos.distanceTo(lastPoint) > 0.05) {
          // Create NEW array with new point (immutable)
          const newTrail = [...existingTrail, pos];
          // Limit trail length
          if (newTrail.length > 500) {
            newTrail.shift();
          }
          newTrails.set(entity.id, newTrail);
        } else {
          // No change needed, keep existing trail
          newTrails.set(entity.id, existingTrail);
        }
      }

      return newTrails;
    });
  }, [state?.tick, state?.run_id]);

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
      <SceneContent state={state} trails={trails} interceptGeometry={interceptGeometry} assignments={assignments} />
    </Canvas>
  );
}
