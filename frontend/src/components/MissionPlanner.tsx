/**
 * Mission Planner - World-Class Interactive Scenario Builder
 *
 * Features:
 * 1. Click-drag to place entity AND set velocity in one gesture
 * 2. Visual gizmo handles for position/velocity manipulation
 * 3. Smooth selection with glow effects and hover states
 * 4. Keyboard shortcuts (Delete, Escape, arrows)
 * 5. Property panel for precise numeric editing
 * 6. Grid snapping and alignment helpers
 * 7. Resizable zones with corner handles
 */

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import type { ThreeEvent } from '@react-three/fiber';
import { Line, Text, Plane, Html } from '@react-three/drei';
import * as THREE from 'three';

// Scale factor matching Scene.tsx
const SCALE = 0.001; // 1 unit = 1km
const INVERSE_SCALE = 1000; // Convert back to meters

export type PlacementMode = 'view' | 'interceptor' | 'target' | 'launcher' | 'zone';

export interface PlannedEntity {
  id: string;
  type: 'interceptor' | 'target' | 'launcher';
  position: { x: number; y: number; z: number }; // In meters
  velocity: { x: number; y: number; z: number }; // In m/s
  color: string;
  // Launcher-specific properties
  launcherConfig?: {
    detectionRange: number;      // meters
    numMissiles: number;
    launchMode: 'auto' | 'manual';
  };
}

export interface PlannedZone {
  id: string;
  name: string;
  center: { x: number; y: number; z: number };
  dimensions: { x: number; y: number; z: number };
  color: string;
}

interface MissionPlannerProps {
  mode: PlacementMode;
  plannedEntities: PlannedEntity[];
  plannedZones: PlannedZone[];
  onAddEntity: (entity: PlannedEntity) => void;
  onUpdateEntity: (id: string, updates: Partial<PlannedEntity>) => void;
  onRemoveEntity: (id: string) => void;
  onAddZone: (zone: PlannedZone) => void;
  onUpdateZone: (id: string, updates: Partial<PlannedZone>) => void;
  onRemoveZone: (id: string) => void;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  showGrid: boolean;
  snapToGrid: boolean;
}

// Color palettes
const INTERCEPTOR_COLORS = ['#3b82f6', '#22c55e', '#06b6d4', '#a855f7', '#f97316'];
const TARGET_COLORS = ['#ef4444', '#f97316', '#dc2626', '#ea580c'];
const LAUNCHER_COLORS = ['#fbbf24', '#f59e0b', '#d97706', '#b45309'];  // Amber/orange for launchers
const ZONE_COLORS = ['#00ff00', '#00ffff', '#ff00ff', '#ffff00'];

// Grid settings
const GRID_SIZE = 0.5; // 500m grid cells

/**
 * Snap a value to grid if enabled
 */
function snapToGridValue(value: number, enabled: boolean): number {
  if (!enabled) return value;
  return Math.round(value / GRID_SIZE) * GRID_SIZE;
}

/**
 * Animated selection ring
 */
function SelectionRing({ position, color, scale = 1 }: { position: [number, number, number]; color: string; scale?: number }) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.5;
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      ringRef.current.scale.setScalar(pulse * scale);
    }
  });

  return (
    <mesh ref={ringRef} position={position} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.28, 0.35, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.9} />
    </mesh>
  );
}

/**
 * Glow effect for selected/hovered entities
 */
function GlowSphere({ position, color, intensity = 0.5, size = 0.5 }: {
  position: [number, number, number];
  color: string;
  intensity?: number;
  size?: number;
}) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshBasicMaterial color={color} transparent opacity={intensity * 0.3} />
    </mesh>
  );
}

/**
 * Draggable velocity handle with large drag plane for smooth interaction
 */
function VelocityHandle({
  position,
  entityPosition,
  onDrag,
  onDragEnd,
  color,
  isHovered,
  onHover,
}: {
  position: [number, number, number];
  entityPosition: [number, number, number];
  onDrag: (point: THREE.Vector3) => void;
  onDragEnd: () => void;
  color: string;
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const handleRef = useRef<THREE.Mesh>(null);
  const { gl, camera } = useThree();

  // Animate handle on hover
  useFrame(() => {
    if (handleRef.current) {
      const targetScale = isHovered || isDragging ? 1.8 : 1.2;
      handleRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.2);
    }
  });

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setIsDragging(true);
    gl.domElement.style.cursor = 'grabbing';
  }, [gl]);

  // Use raycasting against a ground plane for smooth dragging
  const handleGlobalPointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (!isDragging) return;
    e.stopPropagation();

    // Project to ground plane (y=entityPosition.y for same altitude)
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -entityPosition[1]);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (e.nativeEvent.offsetX / gl.domElement.clientWidth) * 2 - 1,
      -(e.nativeEvent.offsetY / gl.domElement.clientHeight) * 2 + 1
    );
    raycaster.setFromCamera(mouse, camera);

    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);

    if (intersection) {
      onDrag(intersection);
    }
  }, [isDragging, entityPosition, gl, camera, onDrag]);

  const handlePointerUp = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (isDragging) {
      e.stopPropagation();
      setIsDragging(false);
      gl.domElement.style.cursor = 'default';
      onDragEnd();
    }
  }, [isDragging, gl, onDragEnd]);

  return (
    <>
      {/* Large invisible drag plane when dragging */}
      {isDragging && (
        <Plane
          args={[50, 50]}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, entityPosition[1], 0]}
          onPointerMove={handleGlobalPointerMove}
          onPointerUp={handlePointerUp}
        >
          <meshBasicMaterial visible={false} />
        </Plane>
      )}

      <group position={position}>
        {/* Large glow when hovered or dragging */}
        {(isHovered || isDragging) && (
          <GlowSphere position={[0, 0, 0]} color="#ffff00" intensity={1.2} size={0.3} />
        )}

        {/* Main handle - larger and more visible */}
        <mesh
          ref={handleRef}
          onPointerDown={handlePointerDown}
          onPointerEnter={() => { onHover(true); gl.domElement.style.cursor = 'grab'; }}
          onPointerLeave={() => { onHover(false); if (!isDragging) gl.domElement.style.cursor = 'default'; }}
        >
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial
            color={isDragging ? '#ffffff' : isHovered ? '#ffff00' : color}
            emissive={isHovered || isDragging ? '#ffff00' : '#000000'}
            emissiveIntensity={isHovered || isDragging ? 0.8 : 0.3}
          />
        </mesh>

        {/* Direction indicator ring */}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.14, 0.18, 16]} />
          <meshBasicMaterial
            color={isDragging ? '#ffffff' : '#ffff00'}
            transparent
            opacity={isHovered || isDragging ? 0.9 : 0.5}
          />
        </mesh>

        {/* Hint text */}
        {isHovered && !isDragging && (
          <Html position={[0.25, 0.15, 0]} style={{ pointerEvents: 'none' }}>
            <div style={{
              background: 'rgba(0,0,0,0.9)',
              color: '#ffff00',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              whiteSpace: 'nowrap',
              fontWeight: 'bold',
              border: '1px solid #ffff00',
            }}>
              DRAG to set direction & speed
            </div>
          </Html>
        )}
      </group>
    </>
  );
}

/**
 * Position move handle with drag plane
 */
function PositionHandle({
  position,
  onDrag,
  onDragEnd,
  color,
  isHovered,
  onHover,
}: {
  position: [number, number, number];
  onDrag: (point: THREE.Vector3) => void;
  onDragEnd: () => void;
  color: string;
  isHovered: boolean;
  onHover: (hovered: boolean) => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const { gl, camera } = useThree();

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setIsDragging(true);
    gl.domElement.style.cursor = 'move';
  }, [gl]);

  const handleGlobalPointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (!isDragging) return;
    e.stopPropagation();

    // Project to ground plane
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -position[1]);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (e.nativeEvent.offsetX / gl.domElement.clientWidth) * 2 - 1,
      -(e.nativeEvent.offsetY / gl.domElement.clientHeight) * 2 + 1
    );
    raycaster.setFromCamera(mouse, camera);

    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);

    if (intersection) {
      onDrag(intersection);
    }
  }, [isDragging, position, gl, camera, onDrag]);

  const handlePointerUp = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (isDragging) {
      e.stopPropagation();
      setIsDragging(false);
      gl.domElement.style.cursor = 'default';
      onDragEnd();
    }
  }, [isDragging, gl, onDragEnd]);

  return (
    <>
      {/* Large drag plane when dragging */}
      {isDragging && (
        <Plane
          args={[50, 50]}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, position[1], 0]}
          onPointerMove={handleGlobalPointerMove}
          onPointerUp={handlePointerUp}
        >
          <meshBasicMaterial visible={false} />
        </Plane>
      )}

      <group position={position}>
        {/* Glow when active */}
        {(isHovered || isDragging) && (
          <GlowSphere position={[0, 0, 0]} color="#00ff00" intensity={1} size={0.25} />
        )}

        <mesh
          onPointerDown={handlePointerDown}
          onPointerEnter={() => { onHover(true); gl.domElement.style.cursor = 'move'; }}
          onPointerLeave={() => { onHover(false); if (!isDragging) gl.domElement.style.cursor = 'default'; }}
        >
          <boxGeometry args={[0.15, 0.15, 0.15]} />
          <meshStandardMaterial
            color={isDragging ? '#ffffff' : isHovered ? '#00ff00' : color}
            emissive={isHovered || isDragging ? '#00ff00' : '#000000'}
            emissiveIntensity={isHovered || isDragging ? 0.6 : 0.2}
          />
        </mesh>

        {/* Cross-hair indicator */}
        <Line
          points={[[-0.2, 0, 0], [0.2, 0, 0]]}
          color={isDragging ? '#ffffff' : '#00ff00'}
          lineWidth={2}
          transparent
          opacity={isHovered || isDragging ? 0.9 : 0.4}
        />
        <Line
          points={[[0, 0, -0.2], [0, 0, 0.2]]}
          color={isDragging ? '#ffffff' : '#00ff00'}
          lineWidth={2}
          transparent
          opacity={isHovered || isDragging ? 0.9 : 0.4}
        />

        {/* Hint text */}
        {isHovered && !isDragging && (
          <Html position={[0.25, 0.15, 0]} style={{ pointerEvents: 'none' }}>
            <div style={{
              background: 'rgba(0,0,0,0.9)',
              color: '#00ff00',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              whiteSpace: 'nowrap',
              fontWeight: 'bold',
              border: '1px solid #00ff00',
            }}>
              DRAG to move position
            </div>
          </Html>
        )}
      </group>
    </>
  );
}

/**
 * Visual marker for a planned entity with full manipulation controls
 */
function EntityMarker({
  entity,
  isSelected,
  isHovered,
  onSelect,
  onHover,
  onUpdatePosition,
  onUpdateVelocity,
  showHandles,
  snapToGrid,
}: {
  entity: PlannedEntity;
  isSelected: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
  onUpdatePosition: (pos: { x: number; y: number; z: number }) => void;
  onUpdateVelocity: (vel: { x: number; y: number; z: number }) => void;
  showHandles: boolean;
  snapToGrid: boolean;
}) {
  const [velocityHandleHovered, setVelocityHandleHovered] = useState(false);
  const [positionHandleHovered, setPositionHandleHovered] = useState(false);
  const markerRef = useRef<THREE.Mesh>(null);
  const { gl } = useThree();

  // Convert to Three.js coordinates
  const position: [number, number, number] = [
    entity.position.x * SCALE,
    entity.position.z * SCALE,
    -entity.position.y * SCALE,
  ];

  // Calculate velocity endpoint for arrow
  const velocityMagnitude = Math.sqrt(
    entity.velocity.x ** 2 + entity.velocity.y ** 2 + entity.velocity.z ** 2
  );

  // Scale velocity for visibility
  const velScale = 0.005;
  const velocityEnd: [number, number, number] = [
    position[0] + entity.velocity.x * velScale,
    position[1] + entity.velocity.z * velScale,
    position[2] - entity.velocity.y * velScale,
  ];

  const isInterceptor = entity.type === 'interceptor';
  const isLauncher = entity.type === 'launcher';

  // Smooth hover animation
  useFrame(() => {
    if (markerRef.current) {
      const targetScale = isHovered && !isSelected ? 1.15 : 1;
      markerRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.15);
    }
  });

  // Handle position drag
  const handlePositionDrag = useCallback((point: THREE.Vector3) => {
    let newX = point.x * INVERSE_SCALE;
    let newY = -point.z * INVERSE_SCALE;

    if (snapToGrid) {
      newX = snapToGridValue(newX / INVERSE_SCALE, true) * INVERSE_SCALE;
      newY = snapToGridValue(newY / INVERSE_SCALE, true) * INVERSE_SCALE;
    }

    onUpdatePosition({
      x: newX,
      y: newY,
      z: entity.position.z,
    });
  }, [entity.position.z, onUpdatePosition, snapToGrid]);

  // Handle velocity drag
  const handleVelocityDrag = useCallback((point: THREE.Vector3) => {
    const entityPos = new THREE.Vector3(position[0], position[1], position[2]);

    const newVel = {
      x: (point.x - entityPos.x) / velScale,
      y: -(point.z - entityPos.z) / velScale,
      z: (point.y - entityPos.y) / velScale,
    };

    // Clamp velocity magnitude
    const mag = Math.sqrt(newVel.x ** 2 + newVel.y ** 2 + newVel.z ** 2);
    const maxVel = 400;
    const minVel = 20;

    if (mag > maxVel) {
      const scale = maxVel / mag;
      newVel.x *= scale;
      newVel.y *= scale;
      newVel.z *= scale;
    } else if (mag < minVel && mag > 0) {
      const scale = minVel / mag;
      newVel.x *= scale;
      newVel.y *= scale;
      newVel.z *= scale;
    }

    onUpdateVelocity(newVel);
  }, [position, onUpdateVelocity]);

  return (
    <group>
      {/* Selection/hover glow */}
      {(isSelected || isHovered) && (
        <GlowSphere
          position={position}
          color={entity.color}
          intensity={isSelected ? 0.8 : 0.4}
          size={0.4}
        />
      )}

      {/* Selection ring */}
      {isSelected && (
        <SelectionRing position={position} color={entity.color} />
      )}

      {/* Entity marker */}
      <mesh
        ref={markerRef}
        position={position}
        rotation={isLauncher ? [-Math.PI / 2, 0, 0] : [0, 0, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerEnter={() => { onHover(true); gl.domElement.style.cursor = 'pointer'; }}
        onPointerLeave={() => { onHover(false); gl.domElement.style.cursor = 'default'; }}
      >
        {isInterceptor ? (
          <coneGeometry args={[0.15, 0.4, 8]} />
        ) : isLauncher ? (
          <cylinderGeometry args={[0.2, 0.25, 0.1, 6]} />
        ) : (
          <sphereGeometry args={[0.18, 16, 16]} />
        )}
        <meshStandardMaterial
          color={entity.color}
          emissive={entity.color}
          emissiveIntensity={isSelected ? 0.6 : isHovered ? 0.4 : 0.2}
        />
      </mesh>

      {/* Entity label */}
      <Text
        position={[position[0], position[1] + 0.5, position[2]]}
        fontSize={0.14}
        color={entity.color}
        anchorX="center"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {entity.id}
      </Text>

      {/* Type label */}
      <Text
        position={[position[0], position[1] + 0.35, position[2]]}
        fontSize={0.08}
        color="#94a3b8"
        anchorX="center"
      >
        {isInterceptor ? 'INTERCEPTOR' : isLauncher ? 'LAUNCHER' : 'TARGET'}
      </Text>

      {/* Launcher detection range circle */}
      {isLauncher && entity.launcherConfig && (
        <group rotation={[-Math.PI / 2, 0, 0]} position={[position[0], 0.01, position[2]]}>
          {/* Detection range fill */}
          <mesh>
            <ringGeometry args={[
              entity.launcherConfig.detectionRange * SCALE * 0.98,
              entity.launcherConfig.detectionRange * SCALE,
              64
            ]} />
            <meshBasicMaterial
              color={entity.color}
              transparent
              opacity={isSelected ? 0.2 : 0.1}
              side={THREE.DoubleSide}
            />
          </mesh>
          {/* Detection range border */}
          <mesh>
            <ringGeometry args={[
              entity.launcherConfig.detectionRange * SCALE - 0.02,
              entity.launcherConfig.detectionRange * SCALE,
              64
            ]} />
            <meshBasicMaterial
              color={entity.color}
              transparent
              opacity={isSelected ? 0.6 : 0.3}
            />
          </mesh>
        </group>
      )}

      {/* Velocity vector arrow - not for launchers (they're stationary) */}
      {!isLauncher && velocityMagnitude > 0 && (
        <>
          {/* Arrow line */}
          <Line
            points={[position, velocityEnd]}
            color={isSelected ? '#ffffff' : entity.color}
            lineWidth={isSelected ? 4 : 2}
          />

          {/* Arrow head */}
          <mesh
            position={velocityEnd}
            rotation={[0, 0, Math.atan2(
              entity.velocity.y,
              entity.velocity.x
            )]}
          >
            <coneGeometry args={[0.06, 0.15, 8]} />
            <meshBasicMaterial color={isSelected ? '#ffffff' : entity.color} />
          </mesh>

          {/* Speed label */}
          <Text
            position={[
              (position[0] + velocityEnd[0]) / 2,
              (position[1] + velocityEnd[1]) / 2 + 0.18,
              (position[2] + velocityEnd[2]) / 2,
            ]}
            fontSize={0.1}
            color={isSelected ? '#ffffff' : entity.color}
            anchorX="center"
            outlineWidth={0.015}
            outlineColor="#000000"
          >
            {`${velocityMagnitude.toFixed(0)} m/s`}
          </Text>
        </>
      )}

      {/* Manipulation handles (when selected) */}
      {showHandles && isSelected && (
        <>
          {/* Position handle (below entity) */}
          <PositionHandle
            position={[position[0], position[1] - 0.25, position[2]]}
            onDrag={handlePositionDrag}
            onDragEnd={() => {}}
            color="#00ff00"
            isHovered={positionHandleHovered}
            onHover={setPositionHandleHovered}
          />

          {/* Velocity handle (at arrow tip) - not for launchers */}
          {!isLauncher && (
            <VelocityHandle
              position={velocityEnd}
              entityPosition={position}
              onDrag={handleVelocityDrag}
              onDragEnd={() => {}}
              color="#ffff00"
              isHovered={velocityHandleHovered}
              onHover={setVelocityHandleHovered}
            />
          )}
        </>
      )}
    </group>
  );
}

/**
 * Trajectory preview - shows predicted path with time markers
 */
function TrajectoryPreview({ entity, showTimeMarkers = true }: { entity: PlannedEntity; showTimeMarkers?: boolean }) {
  const points = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    const duration = 30; // 30 seconds of trajectory
    const steps = 60;
    const dt = duration / steps;

    let x = entity.position.x;
    let y = entity.position.y;
    let z = entity.position.z;

    for (let i = 0; i <= steps; i++) {
      pts.push(new THREE.Vector3(
        x * SCALE,
        z * SCALE,
        -y * SCALE
      ));
      x += entity.velocity.x * dt;
      y += entity.velocity.y * dt;
      z += entity.velocity.z * dt;
    }

    return pts;
  }, [entity.position, entity.velocity]);

  // Time markers every 10 seconds
  const timeMarkers = useMemo(() => {
    if (!showTimeMarkers) return [];
    const markers: { position: THREE.Vector3; time: number }[] = [];
    [10, 20, 30].forEach((time) => {
      const idx = (time / 30) * 60;
      if (points[idx]) {
        markers.push({ position: points[idx], time });
      }
    });
    return markers;
  }, [points, showTimeMarkers]);

  return (
    <group>
      <Line
        points={points}
        color={entity.color}
        lineWidth={1}
        dashed
        dashSize={0.15}
        gapSize={0.08}
        transparent
        opacity={0.6}
      />

      {/* Time markers */}
      {timeMarkers.map(({ position, time }) => (
        <group key={time} position={position}>
          <mesh>
            <sphereGeometry args={[0.04, 8, 8]} />
            <meshBasicMaterial color={entity.color} transparent opacity={0.7} />
          </mesh>
          <Text
            position={[0, 0.12, 0]}
            fontSize={0.07}
            color={entity.color}
            anchorX="center"
          >
            {`${time}s`}
          </Text>
        </group>
      ))}
    </group>
  );
}

/**
 * Zone corner resize handle
 */
function ZoneResizeHandle({
  position,
  corner,
  onDrag,
  onDragEnd,
}: {
  position: [number, number, number];
  corner: string;
  onDrag: (point: THREE.Vector3, corner: string) => void;
  onDragEnd: () => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const { gl } = useThree();

  return (
    <mesh
      position={position}
      onPointerDown={(e) => {
        e.stopPropagation();
        setIsDragging(true);
        gl.domElement.style.cursor = 'nwse-resize';
      }}
      onPointerMove={(e) => {
        if (isDragging) {
          e.stopPropagation();
          onDrag(e.point, corner);
        }
      }}
      onPointerUp={(e) => {
        e.stopPropagation();
        setIsDragging(false);
        gl.domElement.style.cursor = 'default';
        onDragEnd();
      }}
      onPointerEnter={() => { setIsHovered(true); gl.domElement.style.cursor = 'nwse-resize'; }}
      onPointerLeave={() => { setIsHovered(false); if (!isDragging) gl.domElement.style.cursor = 'default'; }}
    >
      <boxGeometry args={[0.1, 0.1, 0.1]} />
      <meshStandardMaterial
        color={isDragging ? '#ffffff' : isHovered ? '#00ff00' : '#88ff88'}
        emissive={isHovered || isDragging ? '#00ff00' : '#000000'}
        emissiveIntensity={isHovered || isDragging ? 0.5 : 0}
      />
    </mesh>
  );
}

/**
 * Planned zone visualization with resize handles
 */
function ZoneMarker({
  zone,
  isSelected,
  isHovered,
  onSelect,
  onHover,
  onUpdateZone,
}: {
  zone: PlannedZone;
  isSelected: boolean;
  isHovered: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
  onUpdateZone: (updates: Partial<PlannedZone>) => void;
}) {
  const { gl } = useThree();

  const position: [number, number, number] = [
    zone.center.x * SCALE,
    zone.center.z * SCALE,
    -zone.center.y * SCALE,
  ];

  const dimensions: [number, number, number] = [
    zone.dimensions.x * SCALE,
    zone.dimensions.z * SCALE,
    zone.dimensions.y * SCALE,
  ];

  // Handle corner resize
  const handleResize = useCallback((point: THREE.Vector3, corner: string) => {
    const currentCenter = new THREE.Vector3(...position);
    const halfW = dimensions[0] / 2;
    const halfD = dimensions[2] / 2;

    let newCenterX = zone.center.x;
    let newCenterY = zone.center.y;
    let newWidth = zone.dimensions.x;
    let newDepth = zone.dimensions.y;

    // Adjust based on which corner is being dragged
    if (corner.includes('x+')) {
      const delta = (point.x - (currentCenter.x + halfW)) * INVERSE_SCALE;
      newWidth = Math.max(200, zone.dimensions.x + delta);
      newCenterX = zone.center.x + delta / 2;
    } else if (corner.includes('x-')) {
      const delta = ((currentCenter.x - halfW) - point.x) * INVERSE_SCALE;
      newWidth = Math.max(200, zone.dimensions.x + delta);
      newCenterX = zone.center.x - delta / 2;
    }

    if (corner.includes('z+')) {
      const delta = (-(point.z) - (currentCenter.z + halfD)) * INVERSE_SCALE;
      newDepth = Math.max(200, zone.dimensions.y + delta);
      newCenterY = zone.center.y + delta / 2;
    } else if (corner.includes('z-')) {
      const delta = ((currentCenter.z - halfD) - (-point.z)) * INVERSE_SCALE;
      newDepth = Math.max(200, zone.dimensions.y + delta);
      newCenterY = zone.center.y - delta / 2;
    }

    onUpdateZone({
      center: { ...zone.center, x: newCenterX, y: newCenterY },
      dimensions: { ...zone.dimensions, x: newWidth, y: newDepth },
    });
  }, [position, dimensions, zone, onUpdateZone]);

  // Corner positions for resize handles
  const corners = [
    { key: 'x+z+', pos: [dimensions[0]/2, 0, -dimensions[2]/2] },
    { key: 'x+z-', pos: [dimensions[0]/2, 0, dimensions[2]/2] },
    { key: 'x-z+', pos: [-dimensions[0]/2, 0, -dimensions[2]/2] },
    { key: 'x-z-', pos: [-dimensions[0]/2, 0, dimensions[2]/2] },
  ];

  return (
    <group position={position}>
      {/* Zone box */}
      <mesh
        onClick={(e) => { e.stopPropagation(); onSelect(); }}
        onPointerEnter={() => { onHover(true); gl.domElement.style.cursor = 'pointer'; }}
        onPointerLeave={() => { onHover(false); gl.domElement.style.cursor = 'default'; }}
      >
        <boxGeometry args={dimensions} />
        <meshBasicMaterial
          color={zone.color}
          transparent
          opacity={isSelected ? 0.35 : isHovered ? 0.25 : 0.15}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Wireframe */}
      <mesh>
        <boxGeometry args={dimensions} />
        <meshBasicMaterial
          color={zone.color}
          wireframe
          transparent
          opacity={isSelected ? 1 : isHovered ? 0.8 : 0.6}
        />
      </mesh>

      {/* Zone label */}
      <Text
        position={[0, dimensions[1] / 2 + 0.2, 0]}
        fontSize={0.14}
        color={zone.color}
        anchorX="center"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {zone.name}
      </Text>

      {/* Dimensions label */}
      <Text
        position={[0, dimensions[1] / 2 + 0.08, 0]}
        fontSize={0.08}
        color="#94a3b8"
        anchorX="center"
      >
        {`${(zone.dimensions.x/1000).toFixed(1)}km x ${(zone.dimensions.y/1000).toFixed(1)}km`}
      </Text>

      {/* Resize handles (when selected) */}
      {isSelected && corners.map(({ key, pos }) => (
        <ZoneResizeHandle
          key={key}
          position={pos as [number, number, number]}
          corner={key}
          onDrag={handleResize}
          onDragEnd={() => {}}
        />
      ))}
    </group>
  );
}

/**
 * Zone drawing preview (while dragging)
 */
function ZoneDrawingPreview({
  startPoint,
  currentPoint,
}: {
  startPoint: THREE.Vector3;
  currentPoint: THREE.Vector3;
}) {
  const center = new THREE.Vector3(
    (startPoint.x + currentPoint.x) / 2,
    0.3,
    (startPoint.z + currentPoint.z) / 2
  );

  const width = Math.abs(currentPoint.x - startPoint.x);
  const depth = Math.abs(currentPoint.z - startPoint.z);
  const height = 0.4;

  if (width < 0.1 || depth < 0.1) return null;

  return (
    <group position={[center.x, center.y, center.z]}>
      <mesh>
        <boxGeometry args={[width, height, depth]} />
        <meshBasicMaterial
          color="#00ff00"
          transparent
          opacity={0.25}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh>
        <boxGeometry args={[width, height, depth]} />
        <meshBasicMaterial color="#00ff00" wireframe transparent opacity={0.9} />
      </mesh>

      {/* Size preview */}
      <Text
        position={[0, height/2 + 0.15, 0]}
        fontSize={0.1}
        color="#00ff00"
        anchorX="center"
      >
        {`${(width * INVERSE_SCALE / 1000).toFixed(1)}km x ${(depth * INVERSE_SCALE / 1000).toFixed(1)}km`}
      </Text>
    </group>
  );
}

/**
 * Entity placement preview (while dragging to set velocity)
 */
function PlacementPreview({
  position,
  velocityEnd,
  type,
  color,
}: {
  position: THREE.Vector3;
  velocityEnd: THREE.Vector3 | null;
  type: 'interceptor' | 'target';
  color: string;
}) {
  const velocity = velocityEnd ? {
    x: (velocityEnd.x - position.x) / 0.005,
    y: -(velocityEnd.z - position.z) / 0.005,
    z: (velocityEnd.y - position.y) / 0.005,
  } : { x: 0, y: 0, z: 0 };

  const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);

  return (
    <group>
      {/* Ghost entity */}
      <mesh position={[position.x, position.y, position.z]}>
        {type === 'interceptor' ? (
          <coneGeometry args={[0.15, 0.4, 8]} />
        ) : (
          <sphereGeometry args={[0.18, 16, 16]} />
        )}
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>

      {/* Velocity vector preview */}
      {velocityEnd && speed > 10 && (
        <>
          <Line
            points={[[position.x, position.y, position.z], [velocityEnd.x, velocityEnd.y, velocityEnd.z]]}
            color={color}
            lineWidth={3}
            transparent
            opacity={0.8}
          />
          <mesh position={[velocityEnd.x, velocityEnd.y, velocityEnd.z]}>
            <coneGeometry args={[0.06, 0.15, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.8} />
          </mesh>

          {/* Speed label */}
          <Text
            position={[
              (position.x + velocityEnd.x) / 2,
              (position.y + velocityEnd.y) / 2 + 0.15,
              (position.z + velocityEnd.z) / 2
            ]}
            fontSize={0.1}
            color={color}
            anchorX="center"
          >
            {`${speed.toFixed(0)} m/s`}
          </Text>
        </>
      )}

      {/* Instruction */}
      <Html position={[position.x, position.y + 0.6, position.z]} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: 'rgba(0,0,0,0.85)',
          color: color,
          padding: '4px 10px',
          borderRadius: '4px',
          fontSize: '11px',
          whiteSpace: 'nowrap',
          border: `1px solid ${color}`,
        }}>
          {velocityEnd ? 'Release to place' : 'Drag to set velocity direction'}
        </div>
      </Html>
    </group>
  );
}

/**
 * Optional grid overlay
 */
function GridOverlay({ visible }: { visible: boolean }) {
  if (!visible) return null;

  const lines = useMemo(() => {
    const result: THREE.Vector3[][] = [];
    const size = 10; // 10km total
    const step = GRID_SIZE;

    for (let i = -size; i <= size; i += step) {
      result.push([
        new THREE.Vector3(i, 0.001, -size),
        new THREE.Vector3(i, 0.001, size),
      ]);
      result.push([
        new THREE.Vector3(-size, 0.001, i),
        new THREE.Vector3(size, 0.001, i),
      ]);
    }

    return result;
  }, []);

  return (
    <group>
      {lines.map((pts, i) => (
        <Line
          key={i}
          points={pts}
          color="#1e293b"
          lineWidth={1}
          transparent
          opacity={0.4}
        />
      ))}
    </group>
  );
}

/**
 * Main Mission Planner scene content
 */
export function MissionPlannerContent({
  mode,
  plannedEntities,
  plannedZones,
  onAddEntity,
  onUpdateEntity,
  onRemoveEntity,
  onAddZone,
  onUpdateZone,
  onRemoveZone,
  selectedEntityId,
  onSelectEntity,
  showGrid,
  snapToGrid,
}: MissionPlannerProps) {
  const [hoveredEntityId, setHoveredEntityId] = useState<string | null>(null);
  const [zoneStart, setZoneStart] = useState<THREE.Vector3 | null>(null);
  const [zoneCurrent, setZoneCurrent] = useState<THREE.Vector3 | null>(null);

  // Placement state (for click-drag entity placement)
  const [placementStart, setPlacementStart] = useState<THREE.Vector3 | null>(null);
  const [placementCurrent, setPlacementCurrent] = useState<THREE.Vector3 | null>(null);
  const [isPlacing, setIsPlacing] = useState(false);

  // Track if hovering over an entity (used to prevent placement when clicking entities)
  const hoveredEntityRef = useRef<string | null>(null);

  const { gl } = useThree();

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Delete selected entity
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedEntityId) {
        e.preventDefault();
        if (selectedEntityId.startsWith('zone_')) {
          onRemoveZone(selectedEntityId);
        } else {
          onRemoveEntity(selectedEntityId);
        }
      }

      // Escape to deselect
      if (e.key === 'Escape') {
        e.preventDefault();
        onSelectEntity(null);
        setPlacementStart(null);
        setPlacementCurrent(null);
        setIsPlacing(false);
        setZoneStart(null);
        setZoneCurrent(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedEntityId, onRemoveEntity, onRemoveZone, onSelectEntity]);

  // Get color for new entity
  const getNextColor = useCallback((type: 'interceptor' | 'target' | 'launcher') => {
    const colors = type === 'interceptor' ? INTERCEPTOR_COLORS :
                   type === 'launcher' ? LAUNCHER_COLORS : TARGET_COLORS;
    const count = plannedEntities.filter(e => e.type === type).length;
    return colors[count % colors.length];
  }, [plannedEntities]);

  // Handle plane click (start placement)
  const handlePlaneClick = useCallback((e: ThreeEvent<MouseEvent>) => {
    // If hovering over an existing entity, don't start placement - let the entity handle the click
    if (hoveredEntityRef.current !== null) {
      return;
    }

    if (mode === 'view') {
      // Deselect if clicking on empty space
      onSelectEntity(null);
      return;
    }

    e.stopPropagation();
    const point = e.point.clone();

    if (mode === 'zone') {
      setZoneStart(point);
      setZoneCurrent(point);
    } else if (mode === 'interceptor' || mode === 'target' || mode === 'launcher') {
      setPlacementStart(point);
      setPlacementCurrent(null);
      setIsPlacing(true);
      gl.domElement.style.cursor = 'crosshair';
    }
  }, [mode, gl, onSelectEntity]);

  // Handle plane drag
  const handlePlaneDrag = useCallback((e: ThreeEvent<PointerEvent>) => {
    const point = e.point.clone();

    if (mode === 'zone' && zoneStart) {
      setZoneCurrent(point);
    } else if (isPlacing && placementStart) {
      setPlacementCurrent(point);
    }
  }, [mode, zoneStart, isPlacing, placementStart]);

  // Handle plane release (complete placement)
  const handlePlaneRelease = useCallback(() => {
    // Complete zone drawing
    if (mode === 'zone' && zoneStart && zoneCurrent) {
      const width = Math.abs(zoneCurrent.x - zoneStart.x) * INVERSE_SCALE;
      const depth = Math.abs(zoneCurrent.z - zoneStart.z) * INVERSE_SCALE;

      if (width > 100 && depth > 100) {
        const centerX = ((zoneStart.x + zoneCurrent.x) / 2) * INVERSE_SCALE;
        const centerY = -((zoneStart.z + zoneCurrent.z) / 2) * INVERSE_SCALE;

        onAddZone({
          id: `zone_${Date.now()}`,
          name: `Zone ${plannedZones.length + 1}`,
          center: { x: centerX, y: centerY, z: 600 },
          dimensions: { x: width, y: depth, z: 400 },
          color: ZONE_COLORS[plannedZones.length % ZONE_COLORS.length],
        });
      }
      setZoneStart(null);
      setZoneCurrent(null);
    }

    // Complete entity placement
    if (isPlacing && placementStart) {
      const type = mode as 'interceptor' | 'target' | 'launcher';
      const simPos = {
        x: snapToGrid
          ? snapToGridValue(placementStart.x, true) * INVERSE_SCALE
          : placementStart.x * INVERSE_SCALE,
        y: snapToGrid
          ? snapToGridValue(-placementStart.z, true) * INVERSE_SCALE
          : -placementStart.z * INVERSE_SCALE,
        z: type === 'launcher' ? 0 : 600,  // Launchers on ground
      };

      // Calculate velocity from drag (launchers are stationary)
      let velocity = type === 'interceptor'
        ? { x: 150, y: 0, z: 20 }  // Default interceptor velocity
        : type === 'launcher'
        ? { x: 0, y: 0, z: 0 }     // Launchers are stationary
        : { x: -100, y: 0, z: 0 }; // Default target velocity

      if (placementCurrent && type !== 'launcher') {
        const velScale = 0.005;
        velocity = {
          x: (placementCurrent.x - placementStart.x) / velScale,
          y: -(placementCurrent.z - placementStart.z) / velScale,
          z: 0,
        };

        // Enforce minimum velocity
        const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2);
        if (speed < 30) {
          // Use default if drag was too short
          velocity = type === 'interceptor'
            ? { x: 150, y: 0, z: 20 }
            : { x: -100, y: 0, z: 0 };
        }

        // Cap maximum velocity
        if (speed > 400) {
          const scale = 400 / speed;
          velocity.x *= scale;
          velocity.y *= scale;
        }
      }

      const id = type === 'interceptor'
        ? `I${plannedEntities.filter(e => e.type === 'interceptor').length + 1}`
        : type === 'launcher'
        ? `B${plannedEntities.filter(e => e.type === 'launcher').length + 1}`
        : `T${plannedEntities.filter(e => e.type === 'target').length + 1}`;

      const entity: PlannedEntity = {
        id,
        type,
        position: simPos,
        velocity,
        color: getNextColor(type),
      };

      // Add launcher-specific config
      if (type === 'launcher') {
        entity.launcherConfig = {
          detectionRange: 5000,
          numMissiles: 4,
          launchMode: 'auto',
        };
      }

      onAddEntity(entity);

      setPlacementStart(null);
      setPlacementCurrent(null);
      setIsPlacing(false);
      gl.domElement.style.cursor = 'default';
    }
  }, [mode, zoneStart, zoneCurrent, isPlacing, placementStart, placementCurrent,
      plannedZones, plannedEntities, onAddZone, onAddEntity, getNextColor, snapToGrid, gl]);

  return (
    <>
      {/* Optional grid */}
      <GridOverlay visible={showGrid} />

      {/* Click/drag plane for placement */}
      {mode !== 'view' && (
        <Plane
          args={[20, 20]}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0.001, 0]}
          onPointerDown={handlePlaneClick}
          onPointerMove={handlePlaneDrag}
          onPointerUp={handlePlaneRelease}
        >
          <meshBasicMaterial visible={false} />
        </Plane>
      )}

      {/* Click plane for deselection in view mode */}
      {mode === 'view' && (
        <Plane
          args={[20, 20]}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, 0.001, 0]}
          onClick={() => onSelectEntity(null)}
        >
          <meshBasicMaterial visible={false} />
        </Plane>
      )}

      {/* Zone drawing preview */}
      {zoneStart && zoneCurrent && (
        <ZoneDrawingPreview startPoint={zoneStart} currentPoint={zoneCurrent} />
      )}

      {/* Entity placement preview */}
      {isPlacing && placementStart && (
        <PlacementPreview
          position={placementStart}
          velocityEnd={placementCurrent}
          type={mode as 'interceptor' | 'target'}
          color={getNextColor(mode as 'interceptor' | 'target')}
        />
      )}

      {/* Planned entities */}
      {plannedEntities.map((entity) => (
        <group key={entity.id}>
          <EntityMarker
            entity={entity}
            isSelected={selectedEntityId === entity.id}
            isHovered={hoveredEntityId === entity.id}
            onSelect={() => onSelectEntity(entity.id)}
            onHover={(hovered) => {
              setHoveredEntityId(hovered ? entity.id : null);
              hoveredEntityRef.current = hovered ? entity.id : null;
            }}
            onUpdatePosition={(pos) => onUpdateEntity(entity.id, { position: pos })}
            onUpdateVelocity={(vel) => onUpdateEntity(entity.id, { velocity: vel })}
            showHandles={mode === 'view'}
            snapToGrid={snapToGrid}
          />
          {/* Trajectory preview - not for launchers */}
          {entity.type !== 'launcher' && <TrajectoryPreview entity={entity} />}
        </group>
      ))}

      {/* Planned zones */}
      {plannedZones.map((zone) => (
        <ZoneMarker
          key={zone.id}
          zone={zone}
          isSelected={selectedEntityId === zone.id}
          isHovered={hoveredEntityId === zone.id}
          onSelect={() => onSelectEntity(zone.id)}
          onHover={(hovered) => {
            setHoveredEntityId(hovered ? zone.id : null);
            hoveredEntityRef.current = hovered ? zone.id : null;
          }}
          onUpdateZone={(updates) => onUpdateZone(zone.id, updates)}
        />
      ))}

      {/* Mode indicator */}
      {mode !== 'view' && (
        <Text
          position={[0, 3, 0]}
          fontSize={0.22}
          color={
            mode === 'interceptor' ? '#3b82f6' :
            mode === 'target' ? '#ef4444' :
            mode === 'launcher' ? '#fbbf24' :
            '#00ff00'
          }
          anchorX="center"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {mode === 'interceptor' ? 'CLICK & DRAG TO PLACE INTERCEPTOR' :
           mode === 'target' ? 'CLICK & DRAG TO PLACE TARGET' :
           mode === 'launcher' ? 'CLICK TO PLACE LAUNCHER' :
           'CLICK & DRAG TO DRAW ZONE'}
        </Text>
      )}
    </>
  );
}

/**
 * Hook for mission planner state management
 */
export function useMissionPlanner() {
  const [mode, setMode] = useState<PlacementMode>('view');
  const [plannedEntities, setPlannedEntities] = useState<PlannedEntity[]>([]);
  const [plannedZones, setPlannedZones] = useState<PlannedZone[]>([]);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [showGrid, setShowGrid] = useState(false);
  const [snapToGrid, setSnapToGrid] = useState(false);

  const addEntity = useCallback((entity: PlannedEntity) => {
    setPlannedEntities(prev => [...prev, entity]);
    setSelectedEntityId(entity.id);
  }, []);

  const updateEntity = useCallback((id: string, updates: Partial<PlannedEntity>) => {
    setPlannedEntities(prev =>
      prev.map(e => e.id === id ? { ...e, ...updates } : e)
    );
  }, []);

  const removeEntity = useCallback((id: string) => {
    setPlannedEntities(prev => prev.filter(e => e.id !== id));
    if (selectedEntityId === id) setSelectedEntityId(null);
  }, [selectedEntityId]);

  const addZone = useCallback((zone: PlannedZone) => {
    setPlannedZones(prev => [...prev, zone]);
    setSelectedEntityId(zone.id);
  }, []);

  const updateZone = useCallback((id: string, updates: Partial<PlannedZone>) => {
    setPlannedZones(prev =>
      prev.map(z => z.id === id ? { ...z, ...updates } : z)
    );
  }, []);

  const removeZone = useCallback((id: string) => {
    setPlannedZones(prev => prev.filter(z => z.id !== id));
    if (selectedEntityId === id) setSelectedEntityId(null);
  }, [selectedEntityId]);

  const clearAll = useCallback(() => {
    setPlannedEntities([]);
    setPlannedZones([]);
    setSelectedEntityId(null);
  }, []);

  const selectEntity = useCallback((id: string | null) => {
    setSelectedEntityId(id);
  }, []);

  return {
    mode,
    setMode,
    plannedEntities,
    plannedZones,
    selectedEntityId,
    showGrid,
    setShowGrid,
    snapToGrid,
    setSnapToGrid,
    addEntity,
    updateEntity,
    removeEntity,
    addZone,
    updateZone,
    removeZone,
    clearAll,
    selectEntity,
  };
}
