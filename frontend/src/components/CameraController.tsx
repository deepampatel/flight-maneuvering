/**
 * CameraController — Tri-view camera system (Sim / Globe / Map).
 *
 * Sim view:
 *   - Free orbit around origin (0,0,0) for flat 3D scene
 *   - LEFT-drag = rotate, RIGHT-drag = pan, scroll = zoom
 *   - Y-up world, no globe constraints
 *
 * Globe view:
 *   - Full orbit around globe center (0,0,0)
 *   - LEFT-drag = rotate, RIGHT-drag = pan, scroll = zoom
 *   - Polar angle clamped to prevent going through globe
 *
 * Map view (Google Maps 2D):
 *   - Camera overhead, north always up
 *   - LEFT-drag = pan, scroll = zoom, no rotation
 *
 * Single OrbitControls instance (always mounted, never unmounted) configured
 * dynamically to avoid state loss and ref invalidation during transitions.
 */

import { useRef, useEffect, useMemo } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { EntityState } from '../types';
import {
  metersToScene, velocityToScene, sceneSurfaceNormal,
  DEFAULT_GLOBE_CONFIG, enuVelocityToGlobe, metersToGlobe,
} from '../utils/globeCoords';
import type { GlobeConfig } from '../utils/globeCoords';

export type CameraMode = 'free' | 'tactical' | 'chase' | 'target' | 'cinematic';
export type ViewMode = 'sim' | 'globe' | 'map';

interface CameraControllerProps {
  mode: CameraMode;
  viewMode: ViewMode;
  entities: EntityState[];
  selectedEntityId: string | null;
  replayProgress?: number;
  globeConfig?: GlobeConfig;
  focusRequest?: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LERP_SPEED = 0.08;
const CHASE_OFFSET = 2.5;
const CHASE_HEIGHT = 1.0;
const TACTICAL_HEIGHT = 8;
const MAP_HEIGHT = 6;
const TRANSITION_DURATION = 0.6; // seconds

// ---------------------------------------------------------------------------
// Helpers — mode-aware coordinate transforms
// ---------------------------------------------------------------------------
function toScenePos(entity: EntityState, viewMode: 'sim' | 'globe', gc: GlobeConfig): THREE.Vector3 {
  const [x, y, z] = metersToScene(entity.position, viewMode, gc);
  return new THREE.Vector3(x, y, z);
}

function getSceneVelocityDir(entity: EntityState, viewMode: 'sim' | 'globe', gc: GlobeConfig): THREE.Vector3 {
  const dir = velocityToScene(entity.position, entity.velocity, viewMode, gc);
  if (dir.lengthSq() < 0.001) return new THREE.Vector3(0, 0, -1);
  return dir.normalize();
}

function getSceneNormal(entity: EntityState, viewMode: 'sim' | 'globe', gc: GlobeConfig): THREE.Vector3 {
  return sceneSurfaceNormal(entity.position, viewMode, gc);
}

function computeEntityBounds(entities: EntityState[], viewMode: 'sim' | 'globe', gc: GlobeConfig) {
  if (entities.length === 0) {
    const [x, y, z] = metersToScene({ x: 0, y: 0, z: 0 }, viewMode, gc);
    return { center: new THREE.Vector3(x, y, z), radius: viewMode === 'sim' ? 5 : 2 };
  }
  const positions = entities.map(e => toScenePos(e, viewMode, gc));
  const center = new THREE.Vector3();
  positions.forEach(p => center.add(p));
  center.divideScalar(positions.length);
  let maxDist = 0;
  positions.forEach(p => { const d = p.distanceTo(center); if (d > maxDist) maxDist = d; });
  return { center, radius: Math.max(maxDist, 0.5) };
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Compute the "north" tangent direction on the globe surface at scenario origin.
 * Used as camera.up in map mode so north is always screen-up.
 */
function globeNorthTangent(gc: GlobeConfig): THREE.Vector3 {
  return enuVelocityToGlobe({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }, gc).normalize();
}

// ---------------------------------------------------------------------------
// CameraController
// ---------------------------------------------------------------------------
export function CameraController({
  mode, viewMode, entities, selectedEntityId, replayProgress,
  globeConfig = DEFAULT_GLOBE_CONFIG, focusRequest,
}: CameraControllerProps) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const targetPosRef = useRef(new THREE.Vector3());
  const lookAtRef = useRef(new THREE.Vector3());
  const initializedRef = useRef(false);
  const gc = globeConfig;

  // The effective 3D viewMode (map uses globe coords on the Three.js side)
  const sceneViewMode: 'sim' | 'globe' = viewMode === 'map' ? 'globe' : viewMode;

  // --- Transition state ---
  const prevViewModeRef = useRef<ViewMode>(viewMode);
  const transitionRef = useRef(0);
  const transitionFromPosRef = useRef(new THREE.Vector3());
  const transitionToPosRef = useRef(new THREE.Vector3());
  const transitionFromLookRef = useRef(new THREE.Vector3());
  const transitionToLookRef = useRef(new THREE.Vector3());
  const transitionFromUpRef = useRef(new THREE.Vector3(0, 1, 0));
  const transitionToUpRef = useRef(new THREE.Vector3(0, 1, 0));
  const isTransitioningRef = useRef(false);

  // Saved camera state per mode (restored when returning)
  const savedSimPosRef = useRef<THREE.Vector3 | null>(null);
  const savedSimLookRef = useRef<THREE.Vector3 | null>(null);
  const savedGlobePosRef = useRef<THREE.Vector3 | null>(null);
  const savedGlobeLookRef = useRef<THREE.Vector3 | null>(null);

  // Focus animation
  const focusAnimRef = useRef(0);
  const focusTargetPosRef = useRef(new THREE.Vector3());
  const focusTargetLookRef = useRef(new THREE.Vector3());
  const prevFocusRequestRef = useRef(focusRequest);

  // Track whether controls config has been applied for current viewMode
  const configAppliedRef = useRef<string | null>(null);

  // --- Computed values ---
  const selectedEntity = useMemo(() => {
    if (!selectedEntityId) return null;
    return entities.find(e => e.id === selectedEntityId) || null;
  }, [entities, selectedEntityId]);

  const firstTarget = useMemo(() => entities.find(e => e.type === 'target') || null, [entities]);
  const firstInterceptor = useMemo(() => entities.find(e => e.type === 'interceptor') || null, [entities]);

  const centroid = useMemo(() => {
    if (entities.length === 0) {
      const [x, y, z] = metersToScene({ x: 0, y: 0, z: 0 }, sceneViewMode, gc);
      return new THREE.Vector3(x, y, z);
    }
    const sum = new THREE.Vector3();
    entities.forEach(e => sum.add(toScenePos(e, sceneViewMode, gc)));
    return sum.divideScalar(entities.length);
  }, [entities, gc, sceneViewMode]);

  // Map mode geometry (only relevant for globe→map transitions)
  const scenarioOriginOnGlobe = useMemo(() => {
    const [x, y, z] = metersToGlobe({ x: 0, y: 0, z: 0 }, gc);
    return new THREE.Vector3(x, y, z);
  }, [gc]);

  const surfaceNormal = useMemo(
    () => scenarioOriginOnGlobe.clone().normalize(),
    [scenarioOriginOnGlobe],
  );

  const northUp = useMemo(() => globeNorthTangent(gc), [gc]);

  const mapDesiredPos = useMemo(
    () => scenarioOriginOnGlobe.clone().add(surfaceNormal.clone().multiplyScalar(MAP_HEIGHT)),
    [scenarioOriginOnGlobe, surfaceNormal],
  );

  const mapDesiredLook = scenarioOriginOnGlobe;

  // --- Configure OrbitControls dynamically ---
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    const configKey = `${viewMode}-${mode}`;
    if (configAppliedRef.current === configKey) return;
    configAppliedRef.current = configKey;

    if (viewMode === 'map') {
      // --- MAP MODE: top-down, LEFT=PAN, no rotation ---
      controls.enableRotate = false;
      controls.enablePan = true;
      controls.enableZoom = true;
      controls.enabled = true;
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.panSpeed = 1.0;
      controls.zoomSpeed = 0.8;
      controls.minDistance = 1;
      controls.maxDistance = 15;
      controls.minPolarAngle = 0;
      controls.maxPolarAngle = Math.PI;
      controls.screenSpacePanning = true;
      controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      };
      controls.touches = {
        ONE: THREE.TOUCH.PAN,
        TWO: THREE.TOUCH.DOLLY_PAN,
      };
    } else if (viewMode === 'sim' && mode === 'free') {
      // --- SIM FREE MODE: orbit around flat scene ---
      controls.enableRotate = true;
      controls.enablePan = true;
      controls.enableZoom = true;
      controls.enabled = true;
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.rotateSpeed = 0.5;
      controls.zoomSpeed = 0.8;
      controls.panSpeed = 0.5;
      controls.minDistance = 0.5;
      controls.maxDistance = 80;
      controls.minPolarAngle = 0.05;
      controls.maxPolarAngle = Math.PI / 2 - 0.05; // can't go below ground
      controls.screenSpacePanning = true;
      controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      };
      controls.touches = {
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN,
      };
    } else if (viewMode === 'globe' && mode === 'free') {
      // --- GLOBE FREE MODE: full orbit ---
      controls.enableRotate = true;
      controls.enablePan = true;
      controls.enableZoom = true;
      controls.enabled = true;
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.rotateSpeed = 0.5;
      controls.zoomSpeed = 0.8;
      controls.panSpeed = 0.3;
      controls.minDistance = gc.globeRadius + 0.3;
      controls.maxDistance = 35;
      controls.minPolarAngle = 0.1;
      controls.maxPolarAngle = Math.PI - 0.1;
      controls.screenSpacePanning = true;
      controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      };
      controls.touches = {
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN,
      };
    } else {
      // --- SCRIPTED MODES (tactical, chase, target, cinematic) ---
      controls.enabled = false;
    }
    controls.update();
  }, [viewMode, mode, gc.globeRadius]);

  // Reset camera mode initialization when camera mode changes
  useEffect(() => { initializedRef.current = false; }, [mode]);

  // --- Detect viewMode change → trigger smooth transition ---
  useEffect(() => {
    if (prevViewModeRef.current !== viewMode) {
      const prevMode = prevViewModeRef.current;

      // Save current camera state for the mode we're leaving
      if (prevMode === 'globe') {
        savedGlobePosRef.current = camera.position.clone();
        savedGlobeLookRef.current = controlsRef.current?.target?.clone()
          || new THREE.Vector3(0, 0, 0);
      } else if (prevMode === 'sim') {
        savedSimPosRef.current = camera.position.clone();
        savedSimLookRef.current = controlsRef.current?.target?.clone()
          || new THREE.Vector3(0, 0, 0);
      }

      // Capture "from" state
      transitionFromPosRef.current.copy(camera.position);
      transitionFromLookRef.current.copy(
        controlsRef.current?.target || new THREE.Vector3(0, 0, 0),
      );
      transitionFromUpRef.current.copy(camera.up);

      // Compute "to" state
      if (viewMode === 'map') {
        transitionToPosRef.current.copy(mapDesiredPos);
        transitionToLookRef.current.copy(mapDesiredLook);
        transitionToUpRef.current.copy(northUp);
      } else if (viewMode === 'globe') {
        transitionToPosRef.current.copy(
          savedGlobePosRef.current || new THREE.Vector3(0, 0, 15),
        );
        transitionToLookRef.current.copy(
          savedGlobeLookRef.current || new THREE.Vector3(0, 0, 0),
        );
        transitionToUpRef.current.set(0, 1, 0);
      } else {
        // sim
        transitionToPosRef.current.copy(
          savedSimPosRef.current || new THREE.Vector3(0, 15, 20),
        );
        transitionToLookRef.current.copy(
          savedSimLookRef.current || new THREE.Vector3(0, 0, 0),
        );
        transitionToUpRef.current.set(0, 1, 0);
      }

      transitionRef.current = TRANSITION_DURATION;
      isTransitioningRef.current = true;
      configAppliedRef.current = null;
      prevViewModeRef.current = viewMode;
    }
  }, [viewMode, camera, mapDesiredPos, mapDesiredLook, northUp]);

  // --- Focus request ---
  useEffect(() => {
    if (focusRequest !== undefined && focusRequest !== prevFocusRequestRef.current) {
      prevFocusRequestRef.current = focusRequest;
      let targetPos: THREE.Vector3;
      let targetLook: THREE.Vector3;

      if (selectedEntity) {
        targetLook = toScenePos(selectedEntity, sceneViewMode, gc);
        const normal = getSceneNormal(selectedEntity, sceneViewMode, gc);
        targetPos = targetLook.clone().add(
          normal.multiplyScalar(viewMode === 'map' ? MAP_HEIGHT : viewMode === 'sim' ? 10 : 3),
        );
      } else {
        const bounds = computeEntityBounds(entities, sceneViewMode, gc);
        targetLook = bounds.center;
        const normal = viewMode === 'sim'
          ? new THREE.Vector3(0, 1, 0)
          : bounds.center.clone().normalize();
        targetPos = targetLook.clone().add(
          normal.multiplyScalar(bounds.radius * 2.5 + 1),
        );
      }
      focusTargetPosRef.current.copy(targetPos);
      focusTargetLookRef.current.copy(targetLook);
      focusAnimRef.current = 0.4;
    }
  }, [focusRequest, selectedEntity, entities, gc, viewMode, sceneViewMode]);

  // =========================================================================
  // Frame loop — runs every render frame
  // =========================================================================
  useFrame((_, delta) => {
    // === Transition animation ===
    if (isTransitioningRef.current && transitionRef.current > 0) {
      if (controlsRef.current) controlsRef.current.enabled = false;

      transitionRef.current = Math.max(0, transitionRef.current - delta);
      const raw = 1 - transitionRef.current / TRANSITION_DURATION;
      const t = easeInOutCubic(Math.min(raw, 1));

      const pos = new THREE.Vector3().lerpVectors(
        transitionFromPosRef.current, transitionToPosRef.current, t,
      );
      const look = new THREE.Vector3().lerpVectors(
        transitionFromLookRef.current, transitionToLookRef.current, t,
      );
      const up = new THREE.Vector3().lerpVectors(
        transitionFromUpRef.current, transitionToUpRef.current, t,
      ).normalize();

      camera.position.copy(pos);
      camera.up.copy(up);
      camera.lookAt(look);

      if (controlsRef.current) {
        controlsRef.current.target.copy(look);
        controlsRef.current.update();
      }

      if (transitionRef.current <= 0) {
        isTransitioningRef.current = false;
        camera.position.copy(transitionToPosRef.current);
        camera.up.copy(transitionToUpRef.current);
        camera.lookAt(transitionToLookRef.current);

        if (controlsRef.current) {
          controlsRef.current.target.copy(transitionToLookRef.current);
          const shouldEnable = viewMode === 'map' || mode === 'free';
          controlsRef.current.enabled = shouldEnable;
          controlsRef.current.update();
        }
        configAppliedRef.current = null;
      }
      return;
    }

    // === Focus animation ===
    if (focusAnimRef.current > 0) {
      focusAnimRef.current = Math.max(0, focusAnimRef.current - delta);
      const raw = 1 - focusAnimRef.current / 0.4;
      const t = easeInOutCubic(Math.min(raw, 1));

      camera.position.lerp(focusTargetPosRef.current, t * 0.2);
      if (controlsRef.current) {
        controlsRef.current.target.lerp(focusTargetLookRef.current, t * 0.2);
        controlsRef.current.update();
      } else {
        camera.lookAt(focusTargetLookRef.current);
      }
      return;
    }

    // === Map mode — keep camera.up aligned to north ===
    if (viewMode === 'map') {
      camera.up.copy(northUp);
      return;
    }

    // === Ensure Y-up for sim and globe modes ===
    camera.up.set(0, 1, 0);

    // === Free mode — OrbitControls handles it ===
    if (mode === 'free') return;

    // === Scripted camera modes (tactical, chase, target, cinematic) ===
    let desiredPos: THREE.Vector3;
    let desiredLookAt: THREE.Vector3;

    if (mode === 'tactical') {
      const normal = viewMode === 'sim'
        ? new THREE.Vector3(0, 1, 0)
        : centroid.clone().normalize();
      desiredPos = centroid.clone().add(normal.multiplyScalar(TACTICAL_HEIGHT));
      desiredLookAt = centroid.clone();
    } else if (mode === 'chase') {
      const entity = selectedEntity || firstInterceptor;
      if (!entity) return;
      const pos = toScenePos(entity, sceneViewMode, gc);
      const dir = getSceneVelocityDir(entity, sceneViewMode, gc);
      const normal = getSceneNormal(entity, sceneViewMode, gc);
      desiredPos = pos.clone()
        .sub(dir.clone().multiplyScalar(CHASE_OFFSET))
        .add(normal.multiplyScalar(CHASE_HEIGHT));
      desiredLookAt = pos.clone().add(dir.clone().multiplyScalar(1));
    } else if (mode === 'target') {
      const entity = firstTarget;
      if (!entity) return;
      const pos = toScenePos(entity, sceneViewMode, gc);
      const dir = getSceneVelocityDir(entity, sceneViewMode, gc);
      const normal = getSceneNormal(entity, sceneViewMode, gc);
      desiredPos = pos.clone()
        .sub(dir.clone().multiplyScalar(CHASE_OFFSET * 0.8))
        .add(normal.multiplyScalar(CHASE_HEIGHT * 0.6));
      desiredLookAt = pos.clone().add(dir.clone().multiplyScalar(1));
    } else if (mode === 'cinematic') {
      const progress = replayProgress || 0;
      if (progress < 0.2) {
        const normal = viewMode === 'sim'
          ? new THREE.Vector3(0, 1, 0)
          : centroid.clone().normalize();
        desiredPos = centroid.clone()
          .add(normal.multiplyScalar(TACTICAL_HEIGHT * 0.6))
          .add(new THREE.Vector3(2, 0, 2));
        desiredLookAt = centroid.clone();
      } else if (progress < 0.6) {
        const entity = firstInterceptor;
        if (!entity) return;
        const pos = toScenePos(entity, sceneViewMode, gc);
        const dir = getSceneVelocityDir(entity, sceneViewMode, gc);
        const normal = getSceneNormal(entity, sceneViewMode, gc);
        desiredPos = pos.clone()
          .sub(dir.clone().multiplyScalar(CHASE_OFFSET))
          .add(normal.multiplyScalar(CHASE_HEIGHT));
        desiredLookAt = pos.clone();
      } else if (progress < 0.8) {
        const entity = firstTarget;
        if (!entity) return;
        const pos = toScenePos(entity, sceneViewMode, gc);
        const dir = getSceneVelocityDir(entity, sceneViewMode, gc);
        const normal = getSceneNormal(entity, sceneViewMode, gc);
        desiredPos = pos.clone()
          .sub(dir.clone().multiplyScalar(CHASE_OFFSET * 0.5))
          .add(normal.multiplyScalar(CHASE_HEIGHT * 0.5));
        desiredLookAt = pos.clone();
      } else {
        const interceptor = firstInterceptor;
        const target = firstTarget;
        if (!interceptor || !target) return;
        const iPos = toScenePos(interceptor, sceneViewMode, gc);
        const tPos = toScenePos(target, sceneViewMode, gc);
        const midpoint = iPos.clone().add(tPos).multiplyScalar(0.5);
        desiredPos = midpoint.clone().add(new THREE.Vector3(0.5, 0.5, 0.5));
        desiredLookAt = midpoint;
      }
    } else {
      return;
    }

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

  // =========================================================================
  // Render — single OrbitControls, always mounted
  // =========================================================================
  return <OrbitControls ref={controlsRef} enableDamping />;
}

// ---------------------------------------------------------------------------
// UI: Camera mode buttons
// ---------------------------------------------------------------------------
interface CameraModeSelectorProps {
  mode: CameraMode;
  onSetMode: (mode: CameraMode) => void;
  hasSelection: boolean;
}

export function CameraModeSelector({ mode, onSetMode, hasSelection }: CameraModeSelectorProps) {
  return (
    <div className="camera-modes">
      <button className={`cam-btn ${mode === 'free' ? 'active' : ''}`} onClick={() => onSetMode('free')} title="Free Camera (1)">
        FREE
      </button>
      <button className={`cam-btn ${mode === 'tactical' ? 'active' : ''}`} onClick={() => onSetMode('tactical')} title="Tactical Overhead (3)">
        TAC
      </button>
      <button className={`cam-btn ${mode === 'chase' ? 'active' : ''}`} onClick={() => onSetMode('chase')}
        title={hasSelection ? 'Chase Selected (2)' : 'Chase Interceptor (2)'}>
        CHASE
      </button>
      <button className={`cam-btn ${mode === 'target' ? 'active' : ''}`} onClick={() => onSetMode('target')} title="Target POV">
        TGT
      </button>
    </div>
  );
}
