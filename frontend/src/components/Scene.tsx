/**
 * 3D Scene — Multi-view visualization of the air defense simulation.
 *
 * Supports three view modes: SIM (flat 3D), GLOBE (Earth sphere), and MAP (Google Maps).
 * All entity positions from the backend (ENU meters) are converted via unified transforms
 * in utils/globeCoords.ts. The backend is completely unchanged.
 */

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Text, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import type { EntityState, SimStateEvent, InterceptGeometry, Vec3, AssignmentResult, SimStateEventWithEnvironment, SensorTrack, EngagementZone, CooperativeState, LauncherState, ProtectedArea, ImpactPrediction, BatteryState, BuilderBatteryConfig, BuilderProtectedArea } from '../types';
import { MissionPlannerContent } from './MissionPlanner';
import { PreviewBatteryPlatform, PreviewProtectedAreaDome } from './ScenePreview';
import type { PlacementMode, PlannedEntity, PlannedZone } from './MissionPlanner';
import { CameraController } from './CameraController';
import type { CameraMode, ViewMode } from './CameraController';
import { MissileMesh } from './models/MissileMesh';
import { AircraftMesh } from './models/AircraftMesh';
import { ExhaustTrail } from './effects/ExhaustTrail';
import { Explosion } from './effects/Explosion';
import { EarthGlobe } from './globe/EarthGlobe';
import { Atmosphere } from './globe/Atmosphere';
import {
  metersToScene, velocityToScene,
  sceneSurfaceQuaternion, metersToSceneLength, offsetFromSurface,
  DEFAULT_GLOBE_CONFIG,
} from '../utils/globeCoords';
import type { GlobeConfig } from '../utils/globeCoords';
import { SceneConfigContext, useSceneConfig } from '../utils/GlobeConfigContext';
import type { SceneConfig } from '../utils/GlobeConfigContext';

// ---------------------------------------------------------------------------
// Color palettes
// ---------------------------------------------------------------------------
const INTERCEPTOR_COLORS = [
  '#3b82f6', '#22c55e', '#06b6d4', '#a855f7',
  '#f97316', '#eab308', '#ec4899', '#14b8a6',
];
const INTERCEPTOR_EMISSIVE = [
  '#1d4ed8', '#15803d', '#0891b2', '#7c3aed',
  '#c2410c', '#a16207', '#be185d', '#0f766e',
];
const TARGET_COLORS = ['#ef4444', '#f97316', '#dc2626', '#ea580c'];
const TARGET_EMISSIVE = ['#991b1b', '#c2410c', '#7f1d1d', '#9a3412'];

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------
interface EntityProps {
  entity: EntityState;
  trail: THREE.Vector3[];
}

interface TargetProps extends EntityProps {
  colorIndex?: number;
  isIntercepted?: boolean;
  isSelected?: boolean;
  onClick?: () => void;
}

// Reusable objects for cone rotation (avoid GC pressure)
const _coneDir = new THREE.Vector3();
const _coneUp = new THREE.Vector3(0, 1, 0);
const _coneQuat = new THREE.Quaternion();
const _coneMatrix = new THREE.Matrix4();
const _coneOrigin = new THREE.Vector3();
const _coneCorrection = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));

// ---------------------------------------------------------------------------
// Shared hooks — reduce per-component useFrame overhead
// ---------------------------------------------------------------------------

/** Orient a group along its velocity vector + optionally spin a selection ring. */
function useVelocityOrientation(
  groupRef: React.RefObject<THREE.Group | null>,
  selectionRef: React.RefObject<THREE.Mesh | null>,
  entityPos: Vec3,
  entityVel: Vec3,
  isSelected: boolean,
) {
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  useFrame((state) => {
    if (groupRef.current) {
      const vel = velocityToScene(entityPos, entityVel, vm, gc);
      if (vel.lengthSq() > 0.001) {
        _coneDir.copy(vel).normalize();
        _coneMatrix.lookAt(_coneOrigin, _coneDir, _coneUp);
        _coneQuat.setFromRotationMatrix(_coneMatrix);
        _coneQuat.multiply(_coneCorrection);
        groupRef.current.quaternion.copy(_coneQuat);
      }
    }
    if (selectionRef.current && isSelected) {
      selectionRef.current.rotation.z = state.clock.elapsedTime * 1.5;
    }
  });
}

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------
function Target({ entity, trail, colorIndex = 0, isIntercepted = false, isSelected = false, onClick }: TargetProps) {
  const groupRef = useRef<THREE.Group>(null);
  const selectionRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const engagementDecision = entity.engagement_decision;
  const color = engagementDecision === 'ignore' ? '#6b7280'
    : engagementDecision === 'track_only' ? '#f59e0b'
    : TARGET_COLORS[colorIndex % TARGET_COLORS.length];
  const emissive = engagementDecision === 'ignore' ? '#374151'
    : engagementDecision === 'track_only' ? '#92400e'
    : TARGET_EMISSIVE[colorIndex % TARGET_EMISSIVE.length];

  const position = metersToScene(entity.position, vm, gc);

  useVelocityOrientation(groupRef, selectionRef, entity.position, entity.velocity, isSelected);

  return (
    <group>
      <group ref={groupRef} position={position} onClick={(e) => { e.stopPropagation(); onClick?.(); }}>
        <AircraftMesh
          color={isIntercepted ? '#6b7280' : color}
          emissive={isIntercepted ? '#374151' : emissive}
          emissiveIntensity={isIntercepted ? 0.1 : isSelected ? 0.6 : 0.3}
          opacity={isIntercepted ? 0.5 : 1}
        />
      </group>

      {isSelected && (
        <mesh ref={selectionRef} position={position}>
          <torusGeometry args={[0.25, 0.015, 8, 32]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.8} />
        </mesh>
      )}

      <Text position={offsetFromSurface(position, 0.3, vm)} fontSize={0.12} color={isIntercepted ? '#6b7280' : color} anchorX="center">
        {entity.id}
      </Text>

      {/* threat_type and engagement_decision labels removed for performance — shown in HUD */}

      {trail.length > 1 && (
        <Line points={trail} color={isIntercepted ? '#9ca3af' : color} lineWidth={3} opacity={0.8} transparent />
      )}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Launcher — TEL (Transporter Erector Launcher) with missile canisters
// ---------------------------------------------------------------------------
function Launcher({ launcher }: { launcher: LauncherState }) {
  const sweepRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const enu = { x: launcher.position.x, y: launcher.position.y, z: 0 };
  const position = metersToScene(enu, vm, gc);
  const quat = sceneSurfaceQuaternion(enu, vm, gc);
  const detectionRange = metersToSceneLength(launcher.detection_range, vm, gc);

  const missilePercent = launcher.missiles_total > 0
    ? launcher.missiles_remaining / launcher.missiles_total : 0;
  const statusColor = missilePercent > 0.5 ? '#22c55e' : missilePercent > 0 ? '#f59e0b' : '#ef4444';

  useFrame((state) => {
    if (sweepRef.current) sweepRef.current.rotation.z = state.clock.elapsedTime * 0.3;
    if (pulseRef.current) {
      const s = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.05;
      pulseRef.current.scale.set(s, s, 1);
    }
  });

  return (
    <group position={position} quaternion={quat}>
      {/* TEL base platform — octagonal concrete pad */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.22, 0.04, 8]} />
        <meshStandardMaterial color="#374151" emissive="#1f2937" emissiveIntensity={0.2} roughness={0.9} metalness={0.1} />
      </mesh>

      {/* Launcher rail — angled upward */}
      <group position={[0, 0.04, 0]} rotation={[0.4, 0, 0]}>
        <mesh position={[0, 0.08, 0]}>
          <boxGeometry args={[0.06, 0.18, 0.04]} />
          <meshStandardMaterial color="#57534e" emissive="#44403c" emissiveIntensity={0.15} roughness={0.6} metalness={0.4} />
        </mesh>
        {/* Missile canisters on the rail */}
        {Array.from({ length: Math.min(launcher.missiles_remaining, 4) }).map((_, i) => (
          <mesh key={i} position={[(i % 2 === 0 ? -0.02 : 0.02), 0.02 + i * 0.04, 0.025]}>
            <cylinderGeometry args={[0.012, 0.012, 0.05, 6]} />
            <meshStandardMaterial color="#a8a29e" emissive="#78716c" emissiveIntensity={0.15} roughness={0.4} metalness={0.6} />
          </mesh>
        ))}
      </group>

      {/* Status ring on ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[0.2, 0.22, 32]} />
        <meshBasicMaterial color={statusColor} transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>

      {/* Detection range — subtle swept sector */}
      <group rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.003, 0]}>
        <mesh ref={sweepRef}>
          <ringGeometry args={[0, detectionRange, 2, 1, 0, 0.04]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.2} side={THREE.DoubleSide} />
        </mesh>
        <mesh ref={pulseRef}>
          <ringGeometry args={[detectionRange * 0.97, detectionRange, 64]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.12} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* Labels */}
      <Text position={[0, 0.35, 0]} fontSize={0.09} color="#fbbf24" anchorX="center" font="https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FPpPFrN5-B_NU.woff2">
        {launcher.id}
      </Text>
      <Text position={[0, 0.25, 0]} fontSize={0.06} color={statusColor} anchorX="center">
        {`${launcher.missiles_remaining}/${launcher.missiles_total}`}
      </Text>

      {/* Tracked target indicators — small diamonds around the launcher */}
      {launcher.tracked_targets && launcher.tracked_targets.map((track, idx) => {
        const angle = (idx / Math.max(launcher.tracked_targets.length, 1)) * Math.PI * 2;
        const engaged = !!track.assigned_interceptor;
        return (
          <mesh key={track.target_id}
            position={[Math.cos(angle) * 0.3, 0.04, Math.sin(angle) * 0.3]}
            rotation={[0, 0, Math.PI / 4]}
          >
            <boxGeometry args={[0.035, 0.035, 0.005]} />
            <meshBasicMaterial color={engaged ? '#22c55e' : '#ef4444'} transparent opacity={0.9} />
          </mesh>
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Interceptor
// ---------------------------------------------------------------------------
interface InterceptorProps extends EntityProps {
  colorIndex?: number;
  isSelected?: boolean;
  onClick?: () => void;
}

function Interceptor({ entity, trail, colorIndex = 0, isSelected = false, onClick }: InterceptorProps) {
  const groupRef = useRef<THREE.Group>(null);
  const selectionRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const color = INTERCEPTOR_COLORS[colorIndex % INTERCEPTOR_COLORS.length];
  const emissive = INTERCEPTOR_EMISSIVE[colorIndex % INTERCEPTOR_EMISSIVE.length];

  const position = metersToScene(entity.position, vm, gc);

  // Scene-frame velocity for exhaust trail
  const sceneVel = useMemo(() => velocityToScene(entity.position, entity.velocity, vm, gc),
    [entity.position, entity.velocity, vm, gc]);
  const velocity: [number, number, number] = [sceneVel.x * 0.001, sceneVel.y * 0.001, sceneVel.z * 0.001];

  useVelocityOrientation(groupRef, selectionRef, entity.position, entity.velocity, isSelected);

  return (
    <group>
      <group ref={groupRef} position={position} onClick={(e) => { e.stopPropagation(); onClick?.(); }}>
        <MissileMesh color={color} emissive={emissive} emissiveIntensity={isSelected ? 0.6 : 0.3} />
      </group>

      <ExhaustTrail position={position} velocity={velocity} color={color} />

      {isSelected && (
        <mesh ref={selectionRef} position={position}>
          <torusGeometry args={[0.22, 0.015, 8, 32]} />
          <meshBasicMaterial color="#fbbf24" transparent opacity={0.8} />
        </mesh>
      )}

      <Text position={offsetFromSurface(position, 0.3, vm)} fontSize={0.12} color={color} anchorX="center">
        {entity.id}
      </Text>

      {trail.length > 1 && <GradientTrail points={trail} color={color} />}
    </group>
  );
}

// ---------------------------------------------------------------------------
// GradientTrail — time-marked trail line
// ---------------------------------------------------------------------------
function GradientTrail({ points, color }: { points: THREE.Vector3[]; color: string }) {
  return <Line points={points} color={color} lineWidth={3} opacity={0.8} transparent />;
}

// ---------------------------------------------------------------------------
// InterceptPointMarker
// ---------------------------------------------------------------------------
function InterceptPointMarker({ point, collision }: { point: Vec3; collision: boolean }) {
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  const position = metersToScene(point, vm, gc);

  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshBasicMaterial color={collision ? '#22c55e' : '#f97316'} wireframe transparent opacity={0.8} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.06, 12, 12]} />
        <meshBasicMaterial color={collision ? '#22c55e' : '#f97316'} transparent opacity={0.4} />
      </mesh>
      <Text position={[0, 0.25, 0]} fontSize={0.1} color={collision ? '#22c55e' : '#f97316'} anchorX="center">
        INTERCEPT
      </Text>
    </group>
  );
}

// ---------------------------------------------------------------------------
// WindIndicator — fixed position near globe
// ---------------------------------------------------------------------------
function WindIndicator({ wind }: { wind: Vec3 | null }) {
  const arrowRef = useRef<THREE.Group>(null);
  const magnitude = wind ? Math.sqrt(wind.x * wind.x + wind.y * wind.y) : 0;

  useFrame(() => {
    if (!arrowRef.current || !wind || magnitude < 0.1) return;
    const angle = Math.atan2(-wind.y, wind.x);
    arrowRef.current.rotation.y = -angle + Math.PI / 2;
  });

  if (!wind || magnitude < 0.1) return null;
  const arrowLength = Math.min(magnitude / 50, 1) * 1.5 + 0.5;

  return (
    <group ref={arrowRef} position={[-7, 7, -7]}>
      <mesh position={[arrowLength / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[0.03, 0.03, arrowLength, 8]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.3} />
      </mesh>
      <mesh position={[arrowLength, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.08, 0.2, 8]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.5} />
      </mesh>
      <Text position={[arrowLength / 2, 0.25, 0]} fontSize={0.1} color="#93c5fd" anchorX="center">
        {`${magnitude.toFixed(0)} m/s`}
      </Text>
      <Text position={[0, -0.2, 0]} fontSize={0.08} color="#60a5fa" anchorX="center">
        WIND
      </Text>
    </group>
  );
}

// ---------------------------------------------------------------------------
// UncertaintyEllipsoid — Kalman filter covariance visualization
// ---------------------------------------------------------------------------
function UncertaintyEllipsoid({ track, color = '#60a5fa' }: { track: SensorTrack; color?: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const pos = metersToScene(track.position, vm, gc);
  const baseRadius = Math.min(Math.max(metersToSceneLength(track.position_uncertainty, vm, gc), 0.03), 0.4);

  const velocityMag = track.velocity
    ? Math.sqrt(track.velocity.x ** 2 + track.velocity.y ** 2 + track.velocity.z ** 2) : 0;
  const alongVelScale = velocityMag > 10 ? 1.3 : 1.0;

  useEffect(() => {
    if (!meshRef.current || !track.velocity || velocityMag < 1) return;
    const dir = velocityToScene(track.position, track.velocity, vm, gc).normalize();
    const quaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4();
    matrix.lookAt(new THREE.Vector3(), dir, _coneUp);
    quaternion.setFromRotationMatrix(matrix);
    meshRef.current.quaternion.copy(quaternion);
    meshRef.current.scale.set(alongVelScale, 1, 1);
  }, [track.velocity, velocityMag, alongVelScale, track.position, gc]);

  const opacity = 0.2 + (track.track_quality || 0.5) * 0.3;

  return (
    <group position={pos}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[baseRadius, 16, 12]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={opacity} />
      </mesh>
      <mesh>
        <sphereGeometry args={[baseRadius * 0.2, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>
      {track.velocity && velocityMag > 5 && (
        <Line
          points={[
            new THREE.Vector3(0, 0, 0),
            velocityToScene(track.position, track.velocity, vm, gc).normalize().multiplyScalar(baseRadius * 1.5),
          ]}
          color={color} lineWidth={1.5} transparent opacity={0.5}
        />
      )}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[baseRadius * 1.1, baseRadius * 1.2, 16]} />
        <meshBasicMaterial color={track.coasting ? '#f97316' : color} transparent opacity={track.coasting ? 0.6 : 0.3} />
      </mesh>
    </group>
  );
}

function TrackUncertainties({ tracks }: { tracks: SensorTrack[] | null }) {
  if (!tracks || tracks.length === 0) return null;
  return (
    <group>
      {tracks.map((track) => {
        if (!track.is_firm) return null;
        return <UncertaintyEllipsoid key={track.track_id} track={track} color={track.coasting ? '#f97316' : '#22c55e'} />;
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Killbox — engagement zone visualization
// ---------------------------------------------------------------------------
function Killbox({ zone }: { zone: EngagementZone }) {
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  const position = metersToScene(zone.center, vm, gc);
  const quat = sceneSurfaceQuaternion(zone.center, vm, gc);
  const dimensions: [number, number, number] = [
    metersToSceneLength(zone.dimensions.x, vm, gc),
    metersToSceneLength(zone.dimensions.z, vm, gc),
    metersToSceneLength(zone.dimensions.y, vm, gc),
  ];
  const rotationY = -zone.rotation * (Math.PI / 180);

  return (
    <group position={position} quaternion={quat}>
      <group rotation={[0, rotationY, 0]}>
        <mesh>
          <boxGeometry args={dimensions} />
          <meshBasicMaterial color={zone.color} transparent opacity={0.15} side={THREE.DoubleSide} />
        </mesh>
        <mesh>
          <boxGeometry args={dimensions} />
          <meshBasicMaterial color={zone.color} wireframe transparent opacity={0.6} />
        </mesh>
        <Text position={[0, dimensions[1] / 2 + 0.15, 0]} fontSize={0.12} color={zone.color} anchorX="center" anchorY="bottom">
          {zone.name}
        </Text>
        <Text position={[0, dimensions[1] / 2 + 0.05, 0]} fontSize={0.08} color={zone.color} anchorX="center" anchorY="top">
          {`P${zone.priority}`}
        </Text>
        {[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([sx, sz], idx) => (
          <mesh key={idx} position={[sx * dimensions[0] / 2, 0, sz * dimensions[2] / 2]}>
            <cylinderGeometry args={[0.02, 0.02, dimensions[1], 8]} />
            <meshBasicMaterial color={zone.color} transparent opacity={0.8} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function EngagementZones({ zones }: { zones: EngagementZone[] | null }) {
  if (!zones || zones.length === 0) return null;
  return (
    <group>
      {zones.filter(zone => zone.active).map((zone) => <Killbox key={zone.zone_id} zone={zone} />)}
    </group>
  );
}

// ---------------------------------------------------------------------------
// HandoffArc — cooperative handoff between interceptors
// ---------------------------------------------------------------------------
function HandoffArc({ fromPos, toPos, status, targetId }: {
  fromPos: Vec3; toPos: Vec3; status: 'pending' | 'approved' | 'executed'; targetId: string;
}) {
  const arcRef = useRef<THREE.Group>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  // Compute arc by interpolating in ENU space and projecting each point
  const arcPoints = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const numPoints = 20;
    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints;
      const interpENU = {
        x: fromPos.x + (toPos.x - fromPos.x) * t,
        y: fromPos.y + (toPos.y - fromPos.y) * t,
        z: (fromPos.z + (toPos.z - fromPos.z) * t) + Math.sin(t * Math.PI) * 500,
      };
      const [gx, gy, gz] = metersToScene(interpENU, vm, gc);
      points.push(new THREE.Vector3(gx, gy, gz));
    }
    return points;
  }, [fromPos, toPos, vm, gc]);

  const color = status === 'pending' ? '#fbbf24' : status === 'approved' ? '#22c55e' : '#60a5fa';

  const [dashOffset, setDashOffset] = useState(0);
  useFrame((_, delta) => {
    if (status === 'pending') setDashOffset(prev => (prev + delta * 2) % 1);
  });

  const from = metersToScene(fromPos, vm, gc);
  const to = metersToScene(toPos, vm, gc);

  return (
    <group ref={arcRef}>
      <Line points={arcPoints} color={color} lineWidth={2}
        dashed={status === 'pending'} dashSize={0.1} gapSize={0.05} opacity={0.8} transparent />
      <mesh position={to} rotation={[0, Math.atan2(to[0] - from[0], to[2] - from[2]), 0]}>
        <coneGeometry args={[0.05, 0.12, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
      <group position={arcPoints[10]}>
        <mesh>
          <sphereGeometry args={[0.06, 12, 12]} />
          <meshBasicMaterial color={color} transparent opacity={0.8} />
        </mesh>
        <Text position={[0, 0.15, 0]} fontSize={0.08} color={color} anchorX="center">
          {status === 'pending' ? 'HANDOFF' : status === 'approved' ? 'APPROVED' : 'DONE'}
        </Text>
        <Text position={[0, 0.05, 0]} fontSize={0.06} color="#94a3b8" anchorX="center">
          {targetId}
        </Text>
      </group>
      {status === 'pending' && (
        <mesh position={arcPoints[10]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.08 + dashOffset * 0.1, 0.1 + dashOffset * 0.1, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.5 * (1 - dashOffset)} />
        </mesh>
      )}
    </group>
  );
}

function HandoffVisualizations({ cooperativeState, entities }: {
  cooperativeState: CooperativeState | null; entities: EntityState[];
}) {
  if (!cooperativeState) return null;
  const handoffs = [
    ...cooperativeState.pending_handoffs.map(h => ({ ...h, status: 'pending' as const })),
    ...(cooperativeState.completed_handoffs || [])
      .filter(h => h.status === 'approved').slice(-3)
      .map(h => ({ ...h, status: 'approved' as const })),
  ];
  if (handoffs.length === 0) return null;
  return (
    <group>
      {handoffs.map((handoff) => {
        const fromEntity = entities.find(e => e.id === handoff.from_interceptor);
        const toEntity = entities.find(e => e.id === handoff.to_interceptor);
        if (!fromEntity || !toEntity) return null;
        return (
          <HandoffArc key={handoff.request_id}
            fromPos={fromEntity.position} toPos={toEntity.position}
            status={handoff.status} targetId={handoff.target_id} />
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------
// LeadPursuitLine
// ---------------------------------------------------------------------------
function LeadPursuitLine({ from, to, collision }: { from: Vec3; to: Vec3; collision: boolean }) {
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  const points = useMemo(() => [
    new THREE.Vector3(...metersToScene(from, vm, gc)),
    new THREE.Vector3(...metersToScene(to, vm, gc)),
  ], [from, to, vm, gc]);

  return (
    <Line points={points} color={collision ? '#22c55e' : '#f97316'}
      lineWidth={1.5} dashed dashSize={0.1} gapSize={0.05} opacity={0.6} transparent />
  );
}

// ---------------------------------------------------------------------------
// PersistentTrail — trails for entities no longer in state
// ---------------------------------------------------------------------------
function PersistentTrail({ points, isTarget, colorIndex }: {
  points: THREE.Vector3[]; isTarget: boolean; colorIndex: number;
}) {
  if (points.length < 2) return null;
  const color = isTarget
    ? TARGET_COLORS[colorIndex % TARGET_COLORS.length]
    : INTERCEPTOR_COLORS[colorIndex % INTERCEPTOR_COLORS.length];
  return (
    <group>
      <Line points={points} color={color} lineWidth={2} opacity={0.7} transparent />
      <mesh position={points[points.length - 1]}>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshBasicMaterial color={color} opacity={0.8} transparent />
      </mesh>
      <mesh position={points[0]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color={color} opacity={0.5} transparent />
      </mesh>
    </group>
  );
}

// ---------------------------------------------------------------------------
// ProtectedAreaDome — green hemisphere over defended zones
// ---------------------------------------------------------------------------
function ProtectedAreaDome({ area }: { area: ProtectedArea }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const enu = { x: area.center.x, y: area.center.y, z: 0 };
  const position = metersToScene(enu, vm, gc);
  const quat = sceneSurfaceQuaternion(enu, vm, gc);
  const radius = metersToSceneLength(area.radius, vm, gc);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.opacity = 0.08 + Math.sin(state.clock.elapsedTime * 0.8) * 0.03;
    }
    if (ringRef.current) ringRef.current.rotation.z = state.clock.elapsedTime * 0.15;
  });

  return (
    <group position={position} quaternion={quat}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#22c55e" transparent opacity={0.1} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.25} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[radius * 0.48, radius * 0.5, 64]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
      <Text position={[0, 0.3, 0]} fontSize={0.18} color="#22c55e" anchorX="center" anchorY="middle">
        {area.name}
      </Text>
      {/* "DEFENDED" label removed for performance — name is sufficient */}
      <Text position={[radius, 0.1, 0]} fontSize={0.08} color="#22c55e" anchorX="center">
        {`${(area.radius / 1000).toFixed(1)}km`}
      </Text>
    </group>
  );
}

// ---------------------------------------------------------------------------
// ImpactPointMarker
// ---------------------------------------------------------------------------
function ImpactPointMarker({ prediction }: { prediction: ImpactPrediction }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const enu = { x: prediction.impact_point.x, y: prediction.impact_point.y, z: 0 };
  const position = metersToScene(enu, vm, gc);
  const quat = sceneSurfaceQuaternion(enu, vm, gc);
  const isThreat = prediction.engage;
  const color = isThreat ? '#ef4444' : '#f59e0b';

  useFrame((state) => {
    if (meshRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.3;
      meshRef.current.scale.set(scale, scale, scale);
    }
    if (ringRef.current) {
      const material = ringRef.current.material as THREE.MeshBasicMaterial;
      material.opacity = 0.2 + Math.sin(state.clock.elapsedTime * 2) * 0.15;
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group position={position} quaternion={quat}>
      <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.08, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <ringGeometry args={[0.15, 0.2, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      <Text position={[0, 0.15, 0]} fontSize={0.08} color={color} anchorX="center">
        {prediction.threat_id}
      </Text>
      <Text position={[0, 0.08, 0]} fontSize={0.06} color={color} anchorX="center">
        {isThreat ? `ENGAGE — ${prediction.area_name || 'AREA'}` : 'NO THREAT'}
      </Text>
      {/* TTI label removed for performance — shown in HUD */}
    </group>
  );
}

// ---------------------------------------------------------------------------
// BatteryPlatform — Defense battery installation with radar dish + status
// ---------------------------------------------------------------------------
function BatteryPlatform({ battery }: { battery: BatteryState }) {
  const radarSweepRef = useRef<THREE.Group>(null);
  const dishRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef<THREE.Mesh>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();

  const enu = { x: battery.position.x, y: battery.position.y, z: 0 };
  const position = metersToScene(enu, vm, gc);
  const quat = sceneSurfaceQuaternion(enu, vm, gc);

  const statusColor = battery.status === 'operational' ? '#22c55e'
    : battery.status === 'degraded' ? '#f59e0b'
    : battery.status === 'winchester' ? '#ef4444' : '#6b7280';

  const tierColor = battery.tier === 'davids_sling' ? '#06b6d4'
    : battery.tier === 'arrow' ? '#8b5cf6' : '#3b82f6';

  const ammoFraction = battery.missiles_total > 0
    ? battery.missiles_remaining / battery.missiles_total : 0;

  const radarRangeScaled = metersToSceneLength(battery.radar_range, vm, gc);
  const sectorAngle = (battery.radar_sector / 360) * Math.PI * 2;

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    // Radar sweep rotation
    if (radarSweepRef.current) radarSweepRef.current.rotation.y = t * 0.6;
    // Dish wobble
    if (dishRef.current) dishRef.current.rotation.y = t * 0.6;
    // Pulse ring
    if (pulseRef.current) {
      const s = 1 + Math.sin(t * 1.5) * 0.08;
      pulseRef.current.scale.set(s, s, 1);
      (pulseRef.current.material as THREE.MeshBasicMaterial).opacity = 0.15 + Math.sin(t * 1.5) * 0.05;
    }
  });

  return (
    <group position={position} quaternion={quat}>
      {/* Ground pad — hexagonal concrete */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.15, 0.18, 0.03, 6]} />
        <meshStandardMaterial color="#1f2937" emissive="#111827" emissiveIntensity={0.1} roughness={0.95} metalness={0} />
      </mesh>

      {/* Status ring around base */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
        <ringGeometry args={[0.16, 0.19, 32]} />
        <meshBasicMaterial color={statusColor} transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>

      {/* Command post — small building */}
      <mesh position={[0, 0.045, 0]}>
        <boxGeometry args={[0.1, 0.06, 0.1]} />
        <meshStandardMaterial color="#374151" emissive="#1f2937" emissiveIntensity={0.15} roughness={0.7} metalness={0.3} />
      </mesh>

      {/* Radar pedestal */}
      <mesh position={[0, 0.1, 0]}>
        <cylinderGeometry args={[0.015, 0.02, 0.06, 6]} />
        <meshStandardMaterial color="#57534e" emissive="#44403c" emissiveIntensity={0.1} roughness={0.5} metalness={0.5} />
      </mesh>

      {/* Radar dish — rotating parabolic dish */}
      <group ref={dishRef} position={[0, 0.14, 0]}>
        <mesh rotation={[0.3, 0, 0]}>
          <sphereGeometry args={[0.05, 12, 8, 0, Math.PI * 2, 0, Math.PI / 3]} />
          <meshStandardMaterial
            color={tierColor}
            emissive={tierColor}
            emissiveIntensity={0.4}
            roughness={0.2}
            metalness={0.7}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Feed horn */}
        <mesh position={[0, 0.02, 0.03]} rotation={[0.3, 0, 0]}>
          <cylinderGeometry args={[0.004, 0.008, 0.025, 4]} />
          <meshStandardMaterial color="#d4d4d8" emissive="#a1a1aa" emissiveIntensity={0.2} />
        </mesh>
      </group>

      {/* Radar sweep cone — rotating sector */}
      <group ref={radarSweepRef} position={[0, 0.004, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        {/* Filled sector */}
        <mesh>
          <ringGeometry args={[0.2, Math.min(radarRangeScaled, 5), 32, 1, 0, sectorAngle]} />
          <meshBasicMaterial color={tierColor} transparent opacity={0.03} side={THREE.DoubleSide} />
        </mesh>
        {/* Leading edge beam */}
        <mesh>
          <ringGeometry args={[0.2, Math.min(radarRangeScaled, 5), 2, 1, 0, 0.02]} />
          <meshBasicMaterial color={tierColor} transparent opacity={0.25} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* Range ring — pulsing */}
      <mesh ref={pulseRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <ringGeometry args={[radarRangeScaled * 0.97, radarRangeScaled, 64]} />
        <meshBasicMaterial color={tierColor} transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>

      {/* Labels */}
      <Text position={[0, 0.25, 0]} fontSize={0.08} color={tierColor} anchorX="center">
        {battery.name}
      </Text>
      <Text position={[0, 0.2, 0]} fontSize={0.04} color={statusColor} anchorX="center">
        {battery.status.toUpperCase()}
      </Text>

      {/* Ammo bar */}
      <group position={[0, 0.17, 0.06]}>
        <mesh>
          <planeGeometry args={[0.16, 0.012]} />
          <meshBasicMaterial color="#1f2937" transparent opacity={0.8} />
        </mesh>
        {ammoFraction > 0 && (
          <mesh position={[-0.08 * (1 - ammoFraction), 0, 0.001]}>
            <planeGeometry args={[0.16 * ammoFraction, 0.008]} />
            <meshBasicMaterial color={ammoFraction > 0.25 ? '#22c55e' : '#ef4444'} transparent opacity={0.9} />
          </mesh>
        )}
      </group>
      <Text position={[0.1, 0.17, 0.06]} fontSize={0.03} color="#9ca3af" anchorX="left">
        {`${battery.missiles_remaining}/${battery.missiles_total}`}
      </Text>

      {/* Active engagement count */}
      {battery.active_engagements > 0 && (
        <Text position={[0, 0.14, 0.06]} fontSize={0.035} color="#f59e0b" anchorX="center">
          {`${battery.active_engagements} ENGAGING`}
        </Text>
      )}

      {/* Tier indicator glow on the ground */}
      <pointLight position={[0, 0.05, 0]} color={tierColor} intensity={0.15} distance={1.5} />
    </group>
  );
}

// ---------------------------------------------------------------------------
// SceneContent — all 3D objects inside the Canvas
// ---------------------------------------------------------------------------
interface ExplosionData {
  id: string;
  position: [number, number, number];
}

interface SceneContentProps {
  state: SimStateEvent | null;
  trails: Map<string, THREE.Vector3[]>;
  interceptGeometry?: InterceptGeometry[] | null;
  assignments?: AssignmentResult | null;
  currentWind?: Vec3 | null;
  sensorTracks?: SensorTrack[] | null;
  cooperativeState?: CooperativeState | null;
  launchers?: LauncherState[] | null;
  cameraMode: CameraMode;
  viewMode: ViewMode;
  selectedEntityId: string | null;
  onSelectEntity?: (id: string | null) => void;
  replayProgress?: number;
  focusRequest?: number;
  explosions?: ExplosionData[];
  onExplosionComplete?: (id: string) => void;
  previewBatteries?: BuilderBatteryConfig[];
  previewProtectedAreas?: BuilderProtectedArea[];
}

function SceneContent({ state, trails, interceptGeometry, assignments, currentWind, sensorTracks, cooperativeState, launchers, cameraMode, viewMode, selectedEntityId, onSelectEntity, replayProgress, focusRequest, explosions, onExplosionComplete, previewBatteries, previewProtectedAreas }: SceneContentProps) {
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  const targets = state?.entities.filter((e) => e.type === 'target') || [];
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];
  const currentEntityIds = new Set(state?.entities.map((e) => e.id) || []);

  const interceptedTargetIds = new Set(state?.intercepted_pairs?.map((pair) => pair[1]) || []);
  const interceptedInterceptorIds = new Set(state?.intercepted_pairs?.map((pair) => pair[0]) || []);

  const assignmentMap = useMemo(() => {
    const map = new Map<string, string>();
    if (assignments?.assignments) {
      for (const a of assignments.assignments) map.set(a.interceptor_id, a.target_id);
    }
    return map;
  }, [assignments]);

  const tierColorMap = useMemo(() => {
    const map = new Map<string, number>();
    if (state?.batteries) {
      for (const bat of state.batteries) {
        for (const logEntry of bat.engagement_log || []) {
          if (logEntry.interceptor_id) {
            const tierOffset = bat.tier === 'davids_sling' ? 2 : bat.tier === 'arrow' ? 3 : 0;
            map.set(logEntry.interceptor_id, tierOffset);
          }
        }
      }
    }
    return map;
  }, [state?.batteries]);

  return (
    <>
      {/* Lighting — adjusted per view mode */}
      <ambientLight intensity={vm === 'sim' ? 0.3 : viewMode === 'map' ? 0.2 : 0.04} />
      <directionalLight position={[10, 10, 5]} intensity={vm === 'sim' ? 0.5 : viewMode === 'map' ? 0.4 : 0.15} color={vm === 'sim' ? '#ffffff' : '#4466aa'} />
      <directionalLight position={[-10, -10, -5]} intensity={vm === 'sim' ? 0.15 : viewMode === 'map' ? 0.2 : 0.08} />

      <CameraController
        mode={cameraMode}
        viewMode={viewMode}
        entities={state?.entities || []}
        selectedEntityId={selectedEntityId}
        replayProgress={replayProgress}
        focusRequest={focusRequest}
      />

      {/* Environment — mode-specific */}
      {vm === 'globe' && (
        <>
          <EarthGlobe radius={gc.globeRadius} />
          <Atmosphere radius={gc.globeRadius} />
        </>
      )}
      {vm === 'sim' && (
        <group>
          <gridHelper args={[40, 40, '#1f2937', '#111827']} />
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
            <planeGeometry args={[100, 100]} />
            <meshStandardMaterial color="#0a0a0f" roughness={1} />
          </mesh>
        </group>
      )}

      {/* Wind Indicator */}
      <WindIndicator wind={currentWind || null} />

      {/* Track Uncertainty Ellipses */}
      <TrackUncertainties tracks={sensorTracks || null} />

      {/* Engagement Zones */}
      <EngagementZones zones={cooperativeState?.engagement_zones || null} />

      {/* Handoff Visualizations */}
      <HandoffVisualizations cooperativeState={cooperativeState || null} entities={state?.entities || []} />

      {/* Preview — builder-configured entities shown before simulation starts */}
      {!state && previewProtectedAreas && previewProtectedAreas.map((area) => (
        <PreviewProtectedAreaDome key={area.id} area={area} />
      ))}
      {!state && previewBatteries && previewBatteries.map((config) => (
        <PreviewBatteryPlatform key={config.id} config={config} />
      ))}

      {/* Protected Area Domes (live) */}
      {state?.protected_areas && state.protected_areas.map((area) => (
        <ProtectedAreaDome key={area.id} area={area} />
      ))}

      {/* Impact Point Markers */}
      {state?.impact_predictions && Object.values(state.impact_predictions).map((pred) => (
        <ImpactPointMarker key={pred.threat_id} prediction={pred as ImpactPrediction} />
      ))}

      {/* Battery Platforms (live) */}
      {state?.batteries && state.batteries.map((battery) => (
        <BatteryPlatform key={battery.id} battery={battery} />
      ))}

      {/* Targets — hide intercepted ones */}
      {targets.filter(t => !interceptedTargetIds.has(t.id)).map((target, idx) => (
        <Target key={target.id} entity={target} trail={trails.get(target.id) || []}
          colorIndex={idx} isIntercepted={false}
          isSelected={selectedEntityId === target.id}
          onClick={() => onSelectEntity?.(selectedEntityId === target.id ? null : target.id)} />
      ))}

      {/* Launchers */}
      {launchers && launchers.map((launcher) => <Launcher key={launcher.id} launcher={launcher} />)}

      {/* Interceptors — hide intercepted, tier-specific colors */}
      {interceptors.filter(i => !interceptedInterceptorIds.has(i.id)).map((interceptor, idx) => {
        const tierIdx = tierColorMap.get(interceptor.id);
        const colorIdx = tierIdx !== undefined ? tierIdx : idx;
        return (
          <Interceptor key={interceptor.id} entity={interceptor} trail={trails.get(interceptor.id) || []}
            colorIndex={colorIdx} isSelected={selectedEntityId === interceptor.id}
            onClick={() => onSelectEntity?.(selectedEntityId === interceptor.id ? null : interceptor.id)} />
        );
      })}

      {/* Persistent trails — entities no longer in state */}
      {Array.from(trails.entries()).map(([entityId, points]) => {
        if (currentEntityIds.has(entityId)) return null;
        const isTarget = entityId.startsWith('T');
        const colorIndex = parseInt(entityId.replace(/\D/g, ''), 10) - 1 || 0;
        return <PersistentTrail key={`trail-${entityId}`} points={points} isTarget={isTarget} colorIndex={colorIndex} />;
      })}

      {/* Intercept Geometry Visualization */}
      {interceptGeometry && interceptGeometry.map((geom) => {
        const interceptor = interceptors.find((i) => i.id === geom.interceptor_id);
        if (!interceptor || !geom.intercept_point) return null;
        if (interceptedInterceptorIds.has(geom.interceptor_id)) return null;
        if (interceptedTargetIds.has(geom.target_id)) return null;
        const assignedTargetId = assignmentMap.get(geom.interceptor_id);
        if (assignedTargetId && geom.target_id !== assignedTargetId) return null;
        const showDetailedMarker = geom.los_range < 3000;

        return (
          <group key={`geom-${geom.interceptor_id}-${geom.target_id}`}>
            {showDetailedMarker && (
              <InterceptPointMarker point={geom.intercept_point} collision={geom.collision_course} />
            )}
            <LeadPursuitLine from={interceptor.position} to={geom.intercept_point} collision={geom.collision_course} />
            {!showDetailedMarker && (
              <mesh position={metersToScene(geom.intercept_point, vm, gc)}>
                <sphereGeometry args={[0.05, 8, 8]} />
                <meshBasicMaterial color={geom.collision_course ? '#22c55e' : '#f97316'} transparent opacity={0.6} />
              </mesh>
            )}
          </group>
        );
      })}

      {/* Explosions */}
      {explosions && explosions.map(exp => (
        <group key={exp.id}>
          <Explosion position={exp.position} onComplete={() => onExplosionComplete?.(exp.id)} />
          <pointLight position={exp.position} color="#ffffff" intensity={2} distance={5} />
        </group>
      ))}

      {/* Ground impact markers */}
      {targets
        .filter(t => !interceptedTargetIds.has(t.id) && t.position.z >= 0 && t.position.z <= 10)
        .map(t => {
          const enu = { x: t.position.x, y: t.position.y, z: 0 };
          const impactPos = metersToScene(enu, vm, gc);
          const impactQuat = sceneSurfaceQuaternion(enu, vm, gc);
          return (
            <group key={`impact-${t.id}`} position={impactPos} quaternion={impactQuat}>
              <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.05, 0.15, 32]} />
                <meshBasicMaterial color="#ff8c00" transparent opacity={0.7} side={THREE.DoubleSide} />
              </mesh>
            </group>
          );
        })}

      {/* Starfield background — hidden in map mode */}
      {viewMode === 'globe' && (
        <Stars radius={100} depth={50} count={2000} factor={6} saturation={0} fade speed={1} />
      )}

      {/* Post-processing bloom — only in globe mode with fewer entities (expensive) */}
      {viewMode === 'globe' && (state?.entities?.length || 0) < 80 && (
        <EffectComposer>
          <Bloom luminanceThreshold={0.7} luminanceSmoothing={0.3} intensity={0.3} mipmapBlur levels={3} />
        </EffectComposer>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// SimulationScene — top-level Canvas wrapper
// ---------------------------------------------------------------------------
interface SimulationSceneProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  interceptGeometry?: InterceptGeometry[] | null;
  assignments?: AssignmentResult | null;
  sensorTracks?: SensorTrack[] | null;
  cooperativeState?: CooperativeState | null;
  launchers?: LauncherState[] | null;
  cameraMode?: CameraMode;
  viewMode?: ViewMode;
  selectedEntityId?: string | null;
  onSelectEntity?: (id: string | null) => void;
  replayProgress?: number;
  focusRequest?: number;
  plannerMode?: PlacementMode;
  plannedEntities?: PlannedEntity[];
  plannedZones?: PlannedZone[];
  onAddEntity?: (entity: PlannedEntity) => void;
  onUpdateEntity?: (id: string, updates: Partial<PlannedEntity>) => void;
  onRemoveEntity?: (id: string) => void;
  onAddZone?: (zone: PlannedZone) => void;
  onUpdateZone?: (id: string, updates: Partial<PlannedZone>) => void;
  onRemoveZone?: (id: string) => void;
  showGrid?: boolean;
  snapToGrid?: boolean;
  onScenePlacement?: (type: 'battery' | 'protected_area', position: { x: number; y: number; z: number }) => void;
  previewBatteries?: BuilderBatteryConfig[];
  previewProtectedAreas?: BuilderProtectedArea[];
  globeConfig?: GlobeConfig;
}

export function SimulationScene({
  state,
  interceptGeometry,
  assignments,
  sensorTracks,
  cooperativeState,
  launchers,
  cameraMode = 'free',
  viewMode = 'globe',
  selectedEntityId = null,
  onSelectEntity,
  replayProgress,
  focusRequest,
  plannerMode = 'view',
  plannedEntities = [],
  plannedZones = [],
  onAddEntity,
  onUpdateEntity,
  onRemoveEntity,
  onAddZone,
  onUpdateZone,
  onRemoveZone,
  showGrid = false,
  snapToGrid = false,
  onScenePlacement,
  previewBatteries,
  previewProtectedAreas,
  globeConfig = DEFAULT_GLOBE_CONFIG,
}: SimulationSceneProps) {
  // Compute the 3D scene viewMode (map uses globe coordinates on Three.js side)
  const sceneViewMode: 'sim' | 'globe' = viewMode === 'map' ? 'globe' : viewMode;
  const sceneConfig = useMemo<SceneConfig>(
    () => ({ viewMode: sceneViewMode, globeConfig }),
    [sceneViewMode, globeConfig],
  );

  // Track previous view mode for trail clearing
  const prevSceneViewModeRef = useRef(sceneViewMode);

  const currentWind = useMemo(() => {
    const envState = state as SimStateEventWithEnvironment;
    return envState?.environment?.current_wind || null;
  }, [state]);

  const MAX_TRAIL_LENGTH = 300;
  const MIN_MOVEMENT_THRESHOLD = 0.02;

  // Trail data stored in ref (mutated in place) — avoids GC pressure from Map/Array recreation
  const trailsRef = useRef<Map<string, THREE.Vector3[]>>(new Map());
  const [trails, setTrails] = useState<Map<string, THREE.Vector3[]>>(new Map());
  const trailTickRef = useRef(0);
  const currentRunIdRef = useRef<string | null>(null);

  // Explosion tracking
  const [explosions, setExplosions] = useState<ExplosionData[]>([]);
  const prevInterceptedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!state?.intercepted_pairs) return;
    const currentPairs = state.intercepted_pairs;
    const newPairs = currentPairs.filter(pair => {
      const key = `${pair[0]}-${pair[1]}`;
      return !prevInterceptedRef.current.has(key);
    });
    if (newPairs.length > 0) {
      const newExplosions: ExplosionData[] = newPairs.map(pair => {
        const interceptor = state.entities.find(e => e.id === pair[0]);
        const target = state.entities.find(e => e.id === pair[1]);
        const entity = interceptor || target;
        const pos: [number, number, number] = entity
          ? metersToScene(entity.position, sceneViewMode, globeConfig)
          : [0, 0, 0];
        return { id: `exp-${pair[0]}-${pair[1]}-${Date.now()}`, position: pos };
      });
      setExplosions(prev => [...prev, ...newExplosions]);
      const updated = new Set(prevInterceptedRef.current);
      newPairs.forEach(pair => updated.add(`${pair[0]}-${pair[1]}`));
      prevInterceptedRef.current = updated;
    }
  }, [state?.intercepted_pairs, state?.entities, globeConfig]);

  useEffect(() => {
    if (state?.run_id && state.run_id !== currentRunIdRef.current) {
      prevInterceptedRef.current = new Set();
      setExplosions([]);
    }
  }, [state?.run_id]);

  const handleExplosionComplete = (id: string) => {
    setExplosions(prev => prev.filter(e => e.id !== id));
  };

  // Clear trails when switching between sim ↔ globe (incompatible coordinate systems)
  useEffect(() => {
    if (prevSceneViewModeRef.current !== sceneViewMode) {
      prevSceneViewModeRef.current = sceneViewMode;
      trailsRef.current.clear();
      setTrails(new Map());
    }
  }, [sceneViewMode]);

  // Build trails — mutate in place to avoid GC pressure, snapshot every 3rd tick
  useEffect(() => {
    if (!state) return;
    const trailMap = trailsRef.current;

    // Reset on new run
    if (state.run_id !== currentRunIdRef.current) {
      currentRunIdRef.current = state.run_id;
      trailMap.clear();
      for (const entity of state.entities) {
        const [gx, gy, gz] = metersToScene(entity.position, sceneViewMode, globeConfig);
        trailMap.set(entity.id, [new THREE.Vector3(gx, gy, gz)]);
      }
      trailTickRef.current = 0;
      setTrails(new Map(trailMap));
      return;
    }

    const deadIds = new Set<string>();
    if (state.intercepted_pairs) {
      for (const pair of state.intercepted_pairs) {
        deadIds.add(pair[0]);
        deadIds.add(pair[1]);
      }
    }

    for (const entity of state.entities) {
      if (deadIds.has(entity.id)) continue;

      const [gx, gy, gz] = metersToScene(entity.position, sceneViewMode, globeConfig);
      let trail = trailMap.get(entity.id);
      if (!trail) {
        trail = [];
        trailMap.set(entity.id, trail);
      }
      const last = trail[trail.length - 1];
      if (!last ||
        Math.abs(gx - last.x) + Math.abs(gy - last.y) + Math.abs(gz - last.z) > MIN_MOVEMENT_THRESHOLD) {
        trail.push(new THREE.Vector3(gx, gy, gz));
        if (trail.length > MAX_TRAIL_LENGTH) {
          trail.splice(0, Math.floor(MAX_TRAIL_LENGTH * 0.1));
        }
      }
    }

    // Only trigger React re-render every 3rd tick
    trailTickRef.current++;
    if (trailTickRef.current % 3 === 0) {
      setTrails(new Map(trailMap));
    }
  }, [state?.tick, state?.run_id, sceneViewMode, globeConfig]);

  const isPlanningMode = plannerMode !== 'view' || (plannedEntities.length > 0 && !state);

  return (
    <Canvas
      frameloop={viewMode === 'map' ? 'never' : 'always'}
      camera={{
        position: [0, 0, 15],
        fov: 45,
        near: 0.1,
        far: 200,
      }}
      style={{ background: viewMode === 'sim' ? '#0d1117' : viewMode === 'map' ? '#0a0a12' : '#030712' }}
    >

      <SceneConfigContext.Provider value={sceneConfig}>
        <SceneContent
          state={state}
          trails={trails}
          interceptGeometry={interceptGeometry}
          assignments={assignments}
          currentWind={currentWind}
          sensorTracks={sensorTracks}
          cooperativeState={cooperativeState}
          launchers={launchers}
          cameraMode={cameraMode}
          viewMode={viewMode}
          selectedEntityId={selectedEntityId}
          onSelectEntity={onSelectEntity}
          replayProgress={replayProgress}
          focusRequest={focusRequest}
          explosions={explosions}
          onExplosionComplete={handleExplosionComplete}
          previewBatteries={previewBatteries}
          previewProtectedAreas={previewProtectedAreas}
        />

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
            onScenePlacement={onScenePlacement}
          />
        )}
      </SceneConfigContext.Provider>
    </Canvas>
  );
}
