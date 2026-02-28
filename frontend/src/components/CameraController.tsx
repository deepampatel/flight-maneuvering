/**
 * CameraController - Multiple camera perspectives for the 3D scene
 *
 * Modes:
 * - free: OrbitControls (default)
 * - tactical: Top-down overhead view
 * - chase: Follow behind selected entity
 * - target: Follow a target entity
 * - cinematic: Auto-switching for replay theater
 */

import { useRef, useEffect, useMemo } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { EntityState } from '../types';

export type CameraMode = 'free' | 'tactical' | 'chase' | 'target' | 'cinematic';

interface CameraControllerProps {
  mode: CameraMode;
  entities: EntityState[];
  selectedEntityId: string | null;
  replayProgress?: number; // 0-1 for cinematic mode
}

const SCALE = 0.001;
const LERP_SPEED = 0.06;
const CHASE_DISTANCE = 2.5; // km behind
const CHASE_HEIGHT = 1.0; // km above
const TACTICAL_HEIGHT = 25; // km above

function toScenePos(pos: { x: number; y: number; z: number }): THREE.Vector3 {
  return new THREE.Vector3(
    pos.x * SCALE,
    pos.z * SCALE,
    -pos.y * SCALE
  );
}

function getEntityVelocityDir(entity: EntityState): THREE.Vector3 {
  const dir = new THREE.Vector3(
    entity.velocity.x,
    entity.velocity.z,
    -entity.velocity.y
  );
  if (dir.lengthSq() < 0.001) {
    return new THREE.Vector3(0, 0, -1);
  }
  return dir.normalize();
}

export function CameraController({ mode, entities, selectedEntityId, replayProgress }: CameraControllerProps) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const targetPosRef = useRef(new THREE.Vector3());
  const lookAtRef = useRef(new THREE.Vector3());
  const initializedRef = useRef(false);

  // Find selected entity
  const selectedEntity = useMemo(() => {
    if (!selectedEntityId) return null;
    return entities.find(e => e.id === selectedEntityId) || null;
  }, [entities, selectedEntityId]);

  // Find first target (for target mode)
  const firstTarget = useMemo(() => {
    return entities.find(e => e.type === 'target') || null;
  }, [entities]);

  // Find first interceptor (for cinematic chase)
  const firstInterceptor = useMemo(() => {
    return entities.find(e => e.type === 'interceptor') || null;
  }, [entities]);

  // Compute centroid of all entities
  const centroid = useMemo(() => {
    if (entities.length === 0) return new THREE.Vector3(0, 0, 0);
    const sum = new THREE.Vector3();
    entities.forEach(e => {
      sum.add(toScenePos(e.position));
    });
    return sum.divideScalar(entities.length);
  }, [entities]);

  // Reset camera when mode changes
  useEffect(() => {
    initializedRef.current = false;
  }, [mode]);

  useFrame(() => {
    if (mode === 'free') return; // OrbitControls handles it

    let desiredPos: THREE.Vector3;
    let desiredLookAt: THREE.Vector3;

    if (mode === 'tactical') {
      // Top-down overhead
      desiredPos = new THREE.Vector3(centroid.x, TACTICAL_HEIGHT, centroid.z);
      desiredLookAt = new THREE.Vector3(centroid.x, 0, centroid.z);
    } else if (mode === 'chase') {
      const entity = selectedEntity || firstInterceptor;
      if (!entity) return;

      const pos = toScenePos(entity.position);
      const dir = getEntityVelocityDir(entity);

      // Position behind entity
      desiredPos = pos.clone()
        .sub(dir.clone().multiplyScalar(CHASE_DISTANCE))
        .add(new THREE.Vector3(0, CHASE_HEIGHT, 0));
      desiredLookAt = pos.clone().add(dir.clone().multiplyScalar(1));
    } else if (mode === 'target') {
      const entity = firstTarget;
      if (!entity) return;

      const pos = toScenePos(entity.position);
      const dir = getEntityVelocityDir(entity);

      desiredPos = pos.clone()
        .sub(dir.clone().multiplyScalar(CHASE_DISTANCE * 0.8))
        .add(new THREE.Vector3(0, CHASE_HEIGHT * 0.6, 0));
      desiredLookAt = pos.clone().add(dir.clone().multiplyScalar(1));
    } else if (mode === 'cinematic') {
      const progress = replayProgress || 0;

      if (progress < 0.2) {
        // Wide establishing shot - tactical view
        desiredPos = new THREE.Vector3(centroid.x, TACTICAL_HEIGHT * 0.6, centroid.z + 5);
        desiredLookAt = centroid.clone();
      } else if (progress < 0.6) {
        // Chase cam on interceptor
        const entity = firstInterceptor;
        if (!entity) return;
        const pos = toScenePos(entity.position);
        const dir = getEntityVelocityDir(entity);
        desiredPos = pos.clone()
          .sub(dir.clone().multiplyScalar(CHASE_DISTANCE))
          .add(new THREE.Vector3(0, CHASE_HEIGHT, 0));
        desiredLookAt = pos.clone();
      } else if (progress < 0.8) {
        // Target POV
        const entity = firstTarget;
        if (!entity) return;
        const pos = toScenePos(entity.position);
        const dir = getEntityVelocityDir(entity);
        desiredPos = pos.clone()
          .sub(dir.clone().multiplyScalar(CHASE_DISTANCE * 0.5))
          .add(new THREE.Vector3(0, CHASE_HEIGHT * 0.5, 0));
        desiredLookAt = pos.clone();
      } else {
        // Close-up of intercept point
        const interceptor = firstInterceptor;
        const target = firstTarget;
        if (!interceptor || !target) return;
        const iPos = toScenePos(interceptor.position);
        const tPos = toScenePos(target.position);
        const midpoint = iPos.clone().add(tPos).multiplyScalar(0.5);
        desiredPos = midpoint.clone().add(new THREE.Vector3(0.5, 0.5, 0.5));
        desiredLookAt = midpoint;
      }
    } else {
      return;
    }

    // Smooth transition
    if (!initializedRef.current) {
      targetPosRef.current.copy(desiredPos!);
      lookAtRef.current.copy(desiredLookAt!);
      initializedRef.current = true;
    } else {
      targetPosRef.current.lerp(desiredPos!, LERP_SPEED);
      lookAtRef.current.lerp(desiredLookAt!, LERP_SPEED);
    }

    camera.position.copy(targetPosRef.current);
    camera.lookAt(lookAtRef.current);
  });

  // Only render OrbitControls in free mode
  if (mode === 'free') {
    return (
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.1}
        maxPolarAngle={Math.PI * 0.48}
      />
    );
  }

  return null;
}

/**
 * CameraModeSelector - Floating UI to switch camera modes
 */
interface CameraModeSelectorProps {
  mode: CameraMode;
  onSetMode: (mode: CameraMode) => void;
  hasSelection: boolean;
}

export function CameraModeSelector({ mode, onSetMode, hasSelection }: CameraModeSelectorProps) {
  return (
    <div className="camera-modes">
      <button
        className={`cam-btn ${mode === 'free' ? 'active' : ''}`}
        onClick={() => onSetMode('free')}
        title="Free Camera"
      >
        FREE
      </button>
      <button
        className={`cam-btn ${mode === 'tactical' ? 'active' : ''}`}
        onClick={() => onSetMode('tactical')}
        title="Tactical Overhead"
      >
        TAC
      </button>
      <button
        className={`cam-btn ${mode === 'chase' ? 'active' : ''}`}
        onClick={() => onSetMode('chase')}
        title={hasSelection ? 'Chase Selected Entity' : 'Chase First Interceptor'}
      >
        CHASE
      </button>
      <button
        className={`cam-btn ${mode === 'target' ? 'active' : ''}`}
        onClick={() => onSetMode('target')}
        title="Target POV"
      >
        TGT
      </button>
    </div>
  );
}
