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
import type { EntityState, SimStateEvent, InterceptGeometry, Vec3, AssignmentResult, SimStateEventWithEnvironment, SensorTrack, EngagementZone, CooperativeState, LauncherState } from '../types';
import { MissionPlannerContent } from './MissionPlanner';
import type { PlacementMode, PlannedEntity, PlannedZone } from './MissionPlanner';

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

/**
 * Launcher visualization - A ground platform that launches interceptors
 * Shows the platform, detection range circle, and missile count
 */
function Launcher({ launcher }: { launcher: LauncherState }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);

  // Convert sim coordinates to Three.js coordinates
  const position: [number, number, number] = [
    launcher.position.x * SCALE,
    launcher.position.z * SCALE, // Z (up) becomes Y in Three.js
    -launcher.position.y * SCALE, // Y (north) becomes -Z
  ];

  // Animate detection range ring
  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.2;
    }
  });

  // Detection range in scene units
  const detectionRange = launcher.detection_range * SCALE;

  // Calculate fill percentage for missile display
  const missilePercent = launcher.missiles_total > 0
    ? launcher.missiles_remaining / launcher.missiles_total
    : 0;

  return (
    <group position={position}>
      {/* Platform base - hexagonal shape */}
      <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.2, 0.25, 0.1, 6]} />
        <meshStandardMaterial
          color="#fbbf24"
          emissive="#b45309"
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Launcher tube on top */}
      <mesh position={[0, 0.12, 0]} rotation={[0.3, 0, 0]}>
        <cylinderGeometry args={[0.05, 0.06, 0.15, 8]} />
        <meshStandardMaterial
          color="#78716c"
          emissive="#44403c"
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* Detection range ring - dashed circle on ground */}
      <group rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <mesh ref={ringRef}>
          <ringGeometry args={[detectionRange * 0.98, detectionRange, 64]} />
          <meshBasicMaterial
            color="#fbbf24"
            transparent
            opacity={0.15}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Outer ring line */}
        <mesh>
          <ringGeometry args={[detectionRange - 0.02, detectionRange, 64]} />
          <meshBasicMaterial
            color="#fbbf24"
            transparent
            opacity={0.4}
          />
        </mesh>
      </group>

      {/* Label */}
      <Text
        position={[0, 0.45, 0]}
        fontSize={0.12}
        color="#fbbf24"
        anchorX="center"
      >
        {launcher.id}
      </Text>

      {/* Missile count indicator */}
      <Text
        position={[0, 0.32, 0]}
        fontSize={0.08}
        color={missilePercent > 0.5 ? '#22c55e' : missilePercent > 0 ? '#f59e0b' : '#ef4444'}
        anchorX="center"
      >
        {`${launcher.missiles_remaining}/${launcher.missiles_total}`}
      </Text>

      {/* Detection range label at edge */}
      <Text
        position={[detectionRange, 0.1, 0]}
        fontSize={0.08}
        color="#fbbf24"
        anchorX="center"
        anchorY="middle"
      >
        {`${(launcher.detection_range / 1000).toFixed(1)}km`}
      </Text>

      {/* Tracked target indicators - small dots for each tracked target */}
      {launcher.tracked_targets && launcher.tracked_targets.map((track, idx) => {
        const angle = (idx / Math.max(launcher.tracked_targets.length, 1)) * Math.PI * 2;
        const indicatorPos: [number, number, number] = [
          Math.cos(angle) * 0.35,
          0.05,
          Math.sin(angle) * 0.35,
        ];
        return (
          <mesh key={track.target_id} position={indicatorPos}>
            <sphereGeometry args={[0.03, 8, 8]} />
            <meshBasicMaterial
              color={track.assigned_interceptor ? '#22c55e' : '#ef4444'}
            />
          </mesh>
        );
      })}
    </group>
  );
}

interface InterceptorProps extends EntityProps {
  colorIndex?: number;
}

// Reusable objects for cone rotation calculation (avoid GC pressure)
const _coneDir = new THREE.Vector3();
const _coneUp = new THREE.Vector3(0, 1, 0);
const _coneQuat = new THREE.Quaternion();
const _coneMatrix = new THREE.Matrix4();
const _coneOrigin = new THREE.Vector3();
const _coneCorrection = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));

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
  // Optimized: reuses static objects instead of creating new ones each frame
  useFrame(() => {
    if (!meshRef.current) return;

    const vel = entity.velocity;
    if (vel.x === 0 && vel.y === 0 && vel.z === 0) return;

    // Reuse direction vector (avoid allocation)
    _coneDir.set(vel.x, vel.z, -vel.y).normalize();

    // Point the cone in the velocity direction
    _coneMatrix.lookAt(_coneOrigin, _coneDir, _coneUp);
    _coneQuat.setFromRotationMatrix(_coneMatrix);

    // Cone points up by default, rotate to point forward
    _coneQuat.multiply(_coneCorrection);

    meshRef.current.quaternion.copy(_coneQuat);
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
 * Wind Indicator - Shows wind direction and magnitude
 * Phase 6: Environmental effects visualization
 */
function WindIndicator({ wind }: { wind: Vec3 | null }) {
  const arrowRef = useRef<THREE.Group>(null);

  // Calculate wind magnitude and direction
  const magnitude = wind ? Math.sqrt(wind.x * wind.x + wind.y * wind.y) : 0;

  // Animate the arrow rotation
  useFrame(() => {
    if (!arrowRef.current || !wind || magnitude < 0.1) return;

    // Calculate direction angle (in XZ plane since Y is up in Three.js)
    const angle = Math.atan2(-wind.y, wind.x); // wind.y is North in sim, -Z in Three.js
    arrowRef.current.rotation.y = -angle + Math.PI / 2;
  });

  // Don't render if no wind
  if (!wind || magnitude < 0.1) return null;

  // Scale arrow length based on wind speed (cap at 50 m/s)
  const arrowLength = Math.min(magnitude / 50, 1) * 1.5 + 0.5;

  return (
    <group ref={arrowRef} position={[-2, 2, -2]}>
      {/* Arrow shaft */}
      <mesh position={[arrowLength / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[0.03, 0.03, arrowLength, 8]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.3} />
      </mesh>

      {/* Arrow head */}
      <mesh position={[arrowLength, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.08, 0.2, 8]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.5} />
      </mesh>

      {/* Wind speed label */}
      <Text
        position={[arrowLength / 2, 0.25, 0]}
        fontSize={0.1}
        color="#93c5fd"
        anchorX="center"
      >
        {`${magnitude.toFixed(0)} m/s`}
      </Text>

      {/* "WIND" label */}
      <Text
        position={[0, -0.2, 0]}
        fontSize={0.08}
        color="#60a5fa"
        anchorX="center"
      >
        WIND
      </Text>
    </group>
  );
}

/**
 * Uncertainty Ellipsoid - Shows track position uncertainty as a 3D ellipsoid
 * Phase 6: Visualizes Kalman filter covariance
 *
 * The ellipsoid dimensions are based on position uncertainty, with
 * velocity uncertainty shown as a separate indicator.
 */
function UncertaintyEllipsoid({
  track,
  color = '#60a5fa',
}: {
  track: SensorTrack;
  color?: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Convert to Three.js coordinates
  const pos: [number, number, number] = [
    track.position.x * SCALE,
    track.position.z * SCALE,
    -track.position.y * SCALE,
  ];

  // Scale uncertainty to scene units (capped for visibility)
  // Use different scales for position uncertainty in different directions
  const baseRadius = Math.min(Math.max(track.position_uncertainty * SCALE, 0.03), 0.4);

  // Create slight asymmetry for 3D effect (elongated in velocity direction)
  const velocityMag = track.velocity ?
    Math.sqrt(track.velocity.x ** 2 + track.velocity.y ** 2 + track.velocity.z ** 2) : 0;
  const alongVelScale = velocityMag > 10 ? 1.3 : 1.0;  // Elongate in velocity direction

  // Rotate ellipsoid to align with velocity if track has velocity
  useEffect(() => {
    if (!meshRef.current || !track.velocity || velocityMag < 1) return;

    const dir = new THREE.Vector3(
      track.velocity.x,
      track.velocity.z,
      -track.velocity.y
    ).normalize();

    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4();

    matrix.lookAt(new THREE.Vector3(), dir, up);
    quaternion.setFromRotationMatrix(matrix);

    meshRef.current.quaternion.copy(quaternion);
    meshRef.current.scale.set(alongVelScale, 1, 1);
  }, [track.velocity, velocityMag, alongVelScale]);

  // Calculate confidence-based opacity (higher confidence = more solid)
  const opacity = 0.2 + (track.track_quality || 0.5) * 0.3;

  return (
    <group position={pos}>
      {/* Main uncertainty ellipsoid */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[baseRadius, 16, 12]} />
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={opacity}
        />
      </mesh>

      {/* Inner core showing track position */}
      <mesh>
        <sphereGeometry args={[baseRadius * 0.2, 8, 8]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Velocity uncertainty vector (if available) */}
      {track.velocity && velocityMag > 5 && (
        <group>
          {/* Velocity direction indicator */}
          <Line
            points={[
              new THREE.Vector3(0, 0, 0),
              new THREE.Vector3(
                track.velocity.x * SCALE * 0.5,
                track.velocity.z * SCALE * 0.5,
                -track.velocity.y * SCALE * 0.5
              ),
            ]}
            color={color}
            lineWidth={1.5}
            transparent
            opacity={0.5}
          />
        </group>
      )}

      {/* Track quality indicator ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[baseRadius * 1.1, baseRadius * 1.2, 16]} />
        <meshBasicMaterial
          color={track.coasting ? '#f97316' : color}
          transparent
          opacity={track.coasting ? 0.6 : 0.3}
        />
      </mesh>
    </group>
  );
}


/**
 * Track Uncertainty Visualization - Shows all track uncertainties
 * Phase 6: Renders uncertainty ellipsoids for tracked targets
 */
function TrackUncertainties({
  tracks,
}: {
  tracks: SensorTrack[] | null;
}) {
  if (!tracks || tracks.length === 0) return null;

  return (
    <group>
      {tracks.map((track) => {
        if (!track.is_firm) return null;
        return (
          <UncertaintyEllipsoid
            key={track.track_id}
            track={track}
            color={track.coasting ? '#f97316' : '#22c55e'}
          />
        );
      })}
    </group>
  );
}

/**
 * Killbox Visualization - Shows engagement zones as 3D boxes
 * Phase 6: Cooperative engagement zone visualization
 */
function Killbox({
  zone,
}: {
  zone: EngagementZone;
}) {
  // Convert sim coordinates to Three.js coordinates
  const position: [number, number, number] = [
    zone.center.x * SCALE,
    zone.center.z * SCALE,  // Z (up) becomes Y in Three.js
    -zone.center.y * SCALE, // Y (north) becomes -Z
  ];

  // Scale dimensions
  const dimensions: [number, number, number] = [
    zone.dimensions.x * SCALE,  // width
    zone.dimensions.z * SCALE,  // height (Z in sim = Y in Three.js)
    zone.dimensions.y * SCALE,  // depth (Y in sim = Z in Three.js)
  ];

  // Rotation (around Y axis in Three.js)
  const rotationY = -zone.rotation * (Math.PI / 180);  // Convert degrees to radians, negate for correct direction

  return (
    <group position={position} rotation={[0, rotationY, 0]}>
      {/* Semi-transparent box */}
      <mesh>
        <boxGeometry args={dimensions} />
        <meshBasicMaterial
          color={zone.color}
          transparent
          opacity={0.15}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Wireframe edges */}
      <mesh>
        <boxGeometry args={dimensions} />
        <meshBasicMaterial
          color={zone.color}
          wireframe
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Zone label at top */}
      <Text
        position={[0, dimensions[1] / 2 + 0.15, 0]}
        fontSize={0.12}
        color={zone.color}
        anchorX="center"
        anchorY="bottom"
      >
        {zone.name}
      </Text>

      {/* Priority indicator */}
      <Text
        position={[0, dimensions[1] / 2 + 0.05, 0]}
        fontSize={0.08}
        color={zone.color}
        anchorX="center"
        anchorY="top"
      >
        {`P${zone.priority}`}
      </Text>

      {/* Corner posts for better 3D perception */}
      {[
        [-1, -1], [-1, 1], [1, -1], [1, 1]
      ].map(([sx, sz], idx) => (
        <mesh
          key={idx}
          position={[
            sx * dimensions[0] / 2,
            0,
            sz * dimensions[2] / 2
          ]}
        >
          <cylinderGeometry args={[0.02, 0.02, dimensions[1], 8]} />
          <meshBasicMaterial color={zone.color} transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Engagement Zones Visualization - Renders all killboxes
 * Phase 6: Cooperative engagement
 */
function EngagementZones({
  zones,
}: {
  zones: EngagementZone[] | null;
}) {
  if (!zones || zones.length === 0) return null;

  return (
    <group>
      {zones
        .filter(zone => zone.active)
        .map((zone) => (
          <Killbox key={zone.zone_id} zone={zone} />
        ))}
    </group>
  );
}

/**
 * Handoff Arc - Shows pending handoff between interceptors
 * Phase 6: Cooperative engagement handoff visualization
 */
function HandoffArc({
  fromPos,
  toPos,
  status,
  targetId,
}: {
  fromPos: Vec3;
  toPos: Vec3;
  status: 'pending' | 'approved' | 'executed';
  targetId: string;
}) {
  const arcRef = useRef<THREE.Group>(null);

  // Convert positions to Three.js coordinates
  const from: [number, number, number] = [
    fromPos.x * SCALE,
    fromPos.z * SCALE + 0.2, // Slightly above
    -fromPos.y * SCALE,
  ];

  const to: [number, number, number] = [
    toPos.x * SCALE,
    toPos.z * SCALE + 0.2,
    -toPos.y * SCALE,
  ];

  // Calculate arc points (parabolic curve between interceptors)
  const arcPoints = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const numPoints = 20;
    const midX = (from[0] + to[0]) / 2;
    const midY = Math.max(from[1], to[1]) + 0.5; // Arc height
    const midZ = (from[2] + to[2]) / 2;

    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;
      // Quadratic bezier curve
      const x = (1 - t) * (1 - t) * from[0] + 2 * (1 - t) * t * midX + t * t * to[0];
      const y = (1 - t) * (1 - t) * from[1] + 2 * (1 - t) * t * midY + t * t * to[1];
      const z = (1 - t) * (1 - t) * from[2] + 2 * (1 - t) * t * midZ + t * t * to[2];
      points.push(new THREE.Vector3(x, y, z));
    }
    return points;
  }, [from, to]);

  // Status-based colors
  const color = status === 'pending' ? '#fbbf24' :
                status === 'approved' ? '#22c55e' : '#60a5fa';

  // Animate pending handoffs
  const [dashOffset, setDashOffset] = useState(0);
  useFrame((_, delta) => {
    if (status === 'pending') {
      setDashOffset(prev => (prev + delta * 2) % 1);
    }
  });

  return (
    <group ref={arcRef}>
      {/* Arc line */}
      <Line
        points={arcPoints}
        color={color}
        lineWidth={2}
        dashed={status === 'pending'}
        dashSize={0.1}
        gapSize={0.05}
        opacity={0.8}
        transparent
      />

      {/* Arrow head at destination */}
      <mesh position={to} rotation={[0, Math.atan2(to[0] - from[0], to[2] - from[2]), 0]}>
        <coneGeometry args={[0.05, 0.12, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>

      {/* Status indicator at midpoint */}
      <group position={arcPoints[10]}>
        <mesh>
          <sphereGeometry args={[0.06, 12, 12]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
        <Text
          position={[0, 0.15, 0]}
          fontSize={0.08}
          color={color}
          anchorX="center"
        >
          {status === 'pending' ? 'HANDOFF' : status === 'approved' ? 'APPROVED' : 'DONE'}
        </Text>
        <Text
          position={[0, 0.05, 0]}
          fontSize={0.06}
          color="#94a3b8"
          anchorX="center"
        >
          {targetId}
        </Text>
      </group>

      {/* Pulsing ring for pending */}
      {status === 'pending' && (
        <mesh position={arcPoints[10]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.08 + dashOffset * 0.1, 0.1 + dashOffset * 0.1, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.5 * (1 - dashOffset)} />
        </mesh>
      )}
    </group>
  );
}

/**
 * Handoff Visualizations - Shows all pending/active handoffs
 * Phase 6: Cooperative engagement
 */
function HandoffVisualizations({
  cooperativeState,
  entities,
}: {
  cooperativeState: CooperativeState | null;
  entities: EntityState[];
}) {
  if (!cooperativeState) return null;

  const handoffs = [
    ...cooperativeState.pending_handoffs.map(h => ({ ...h, status: 'pending' as const })),
    ...(cooperativeState.completed_handoffs || [])
      .filter(h => h.status === 'approved')
      .slice(-3) // Show last 3 approved
      .map(h => ({ ...h, status: 'approved' as const })),
  ];

  if (handoffs.length === 0) return null;

  return (
    <group>
      {handoffs.map((handoff) => {
        // Find interceptor positions
        const fromEntity = entities.find(e => e.id === handoff.from_interceptor);
        const toEntity = entities.find(e => e.id === handoff.to_interceptor);

        if (!fromEntity || !toEntity) return null;

        return (
          <HandoffArc
            key={handoff.request_id}
            fromPos={fromEntity.position}
            toPos={toEntity.position}
            status={handoff.status}
            targetId={handoff.target_id}
          />
        );
      })}
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
  currentWind?: Vec3 | null;
  sensorTracks?: SensorTrack[] | null;  // Phase 6: Track uncertainty visualization
  cooperativeState?: CooperativeState | null;  // Phase 6: Cooperative engagement
  launchers?: LauncherState[] | null;  // Launch platforms
}

function SceneContent({ state, trails, interceptGeometry, assignments, currentWind, sensorTracks, cooperativeState, launchers }: SceneContentProps) {
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

      {/* Wind Indicator - Phase 6 */}
      <WindIndicator wind={currentWind || null} />

      {/* Track Uncertainty Ellipses - Phase 6 */}
      <TrackUncertainties tracks={sensorTracks || null} />

      {/* Engagement Zones (Killboxes) - Phase 6 Cooperative */}
      <EngagementZones zones={cooperativeState?.engagement_zones || null} />

      {/* Handoff Visualizations - Phase 6 Cooperative */}
      <HandoffVisualizations
        cooperativeState={cooperativeState || null}
        entities={state?.entities || []}
      />

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

      {/* All Launchers - Launch platforms */}
      {launchers && launchers.map((launcher) => (
        <Launcher key={launcher.id} launcher={launcher} />
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

      {/* Intercept Geometry Visualization - Only show for assigned targets with LOD */}
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

        // LOD: Skip rendering detailed markers for distant intercept points
        // Only show full markers if intercept is within 3km, otherwise just the line
        const distanceToIntercept = geom.los_range; // meters
        const showDetailedMarker = distanceToIntercept < 3000;

        return (
          <group key={`geom-${geom.interceptor_id}-${geom.target_id}`}>
            {/* Intercept point marker - LOD: only show when close */}
            {showDetailedMarker && (
              <InterceptPointMarker
                point={geom.intercept_point}
                collision={geom.collision_course}
              />
            )}
            {/* Lead pursuit line from interceptor to intercept point */}
            <LeadPursuitLine
              from={interceptor.position}
              to={geom.intercept_point}
              collision={geom.collision_course}
            />
            {/* Distant marker - simpler indicator when far */}
            {!showDetailedMarker && (
              <mesh
                position={[
                  geom.intercept_point.x * SCALE,
                  geom.intercept_point.z * SCALE,
                  -geom.intercept_point.y * SCALE,
                ]}
              >
                <sphereGeometry args={[0.05, 8, 8]} />
                <meshBasicMaterial
                  color={geom.collision_course ? '#22c55e' : '#f97316'}
                  transparent
                  opacity={0.6}
                />
              </mesh>
            )}
          </group>
        );
      })}
    </>
  );
}

interface SimulationSceneProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  interceptGeometry?: InterceptGeometry[] | null;
  assignments?: AssignmentResult | null;
  sensorTracks?: SensorTrack[] | null;  // Phase 6
  cooperativeState?: CooperativeState | null;  // Phase 6: Cooperative engagement
  launchers?: LauncherState[] | null;  // Launch platforms
  // Mission Planner props
  plannerMode?: PlacementMode;
  plannedEntities?: PlannedEntity[];
  plannedZones?: PlannedZone[];
  onAddEntity?: (entity: PlannedEntity) => void;
  onUpdateEntity?: (id: string, updates: Partial<PlannedEntity>) => void;
  onRemoveEntity?: (id: string) => void;
  onAddZone?: (zone: PlannedZone) => void;
  onUpdateZone?: (id: string, updates: Partial<PlannedZone>) => void;
  onRemoveZone?: (id: string) => void;
  selectedEntityId?: string | null;
  onSelectEntity?: (id: string | null) => void;
  showGrid?: boolean;
  snapToGrid?: boolean;
}

export function SimulationScene({
  state,
  interceptGeometry,
  assignments,
  sensorTracks,
  cooperativeState,
  launchers,
  // Mission Planner
  plannerMode = 'view',
  plannedEntities = [],
  plannedZones = [],
  onAddEntity,
  onUpdateEntity,
  onRemoveEntity,
  onAddZone,
  onUpdateZone,
  onRemoveZone,
  selectedEntityId = null,
  onSelectEntity,
  showGrid = false,
  snapToGrid = false,
}: SimulationSceneProps) {
  // Extract current wind from environment state if available
  const currentWind = useMemo(() => {
    const envState = state as SimStateEventWithEnvironment;
    return envState?.environment?.current_wind || null;
  }, [state]);
  // Trail configuration
  const MAX_TRAIL_LENGTH = 500;
  const MIN_MOVEMENT_THRESHOLD = 0.02; // Reduced threshold for better trail visibility

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

      // Create new Map for immutable update
      const newTrails = new Map<string, THREE.Vector3[]>();

      // Copy existing trails that aren't in current entities (for persistence)
      for (const [entityId, trail] of prevTrails) {
        if (!state.entities.find(e => e.id === entityId)) {
          newTrails.set(entityId, trail);
        }
      }

      // Update trails for current entities
      for (const entity of state.entities) {
        const pos = new THREE.Vector3(
          entity.position.x * SCALE,
          entity.position.z * SCALE,
          -entity.position.y * SCALE
        );

        const existingTrail = prevTrails.get(entity.id) || [];
        const lastPoint = existingTrail[existingTrail.length - 1];

        // Add point if moved enough (avoid cluttering with tiny movements)
        if (!lastPoint || pos.distanceTo(lastPoint) > MIN_MOVEMENT_THRESHOLD) {
          // Create new array with new point
          const newTrail = [...existingTrail, pos];

          // Efficient trail trimming: trim 10% at once when exceeding limit
          if (newTrail.length > MAX_TRAIL_LENGTH) {
            const trimAmount = Math.floor(MAX_TRAIL_LENGTH * 0.1);
            newTrails.set(entity.id, newTrail.slice(trimAmount));
          } else {
            newTrails.set(entity.id, newTrail);
          }
        } else {
          // No change needed, keep existing trail
          newTrails.set(entity.id, existingTrail);
        }
      }

      return newTrails;
    });
  }, [state?.tick, state?.run_id]);

  // Check if we're in planning mode (no simulation running)
  const isPlanningMode = plannerMode !== 'view' || (plannedEntities.length > 0 && !state);

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
      <SceneContent
        state={state}
        trails={trails}
        interceptGeometry={interceptGeometry}
        assignments={assignments}
        currentWind={currentWind}
        sensorTracks={sensorTracks}
        cooperativeState={cooperativeState}
        launchers={launchers}
      />

      {/* Mission Planner overlay */}
      {isPlanningMode && onAddEntity && onUpdateEntity && onRemoveEntity && onAddZone && onUpdateZone && onRemoveZone && onSelectEntity && (
        <MissionPlannerContent
          mode={plannerMode}
          plannedEntities={plannedEntities}
          plannedZones={plannedZones}
          onAddEntity={onAddEntity}
          onUpdateEntity={onUpdateEntity}
          onRemoveEntity={onRemoveEntity}
          onAddZone={onAddZone}
          onUpdateZone={onUpdateZone}
          onRemoveZone={onRemoveZone}
          selectedEntityId={selectedEntityId}
          onSelectEntity={onSelectEntity}
          showGrid={showGrid}
          snapToGrid={snapToGrid}
        />
      )}
    </Canvas>
  );
}
