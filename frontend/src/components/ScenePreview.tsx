/**
 * ScenePreview — Ghost preview components for builder-configured entities
 *
 * These render batteries and protected areas in the 3D scene BEFORE the
 * simulation starts, so the user can see what they've configured.
 * Uses unified scene coordinates via the SceneConfigContext from Scene.tsx.
 */

import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { TIER_DEFAULTS } from '../data/tierDefaults';
import type { BuilderBatteryConfig, BuilderProtectedArea } from '../types';
import { metersToScene, sceneSurfaceQuaternion, metersToSceneLength } from '../utils/globeCoords';
import { useSceneConfig } from '../utils/GlobeConfigContext';

/**
 * PreviewBatteryPlatform — Ghost battery shown before launch
 */
export function PreviewBatteryPlatform({ config }: { config: BuilderBatteryConfig }) {
  const radarRef = useRef<THREE.Group>(null);
  const { viewMode: vm, globeConfig: gc } = useSceneConfig();
  const tier = TIER_DEFAULTS[config.tier];
  const color = tier?.color || '#3b82f6';

  const enu = { x: config.position.x, y: config.position.y, z: 0 };
  const position = metersToScene(enu, vm, gc);
  const quat = sceneSurfaceQuaternion(enu, vm, gc);
  const radarRangeScaled = metersToSceneLength(config.radar_range, vm, gc);
  const sectorAngle = (config.radar_sector / 360) * Math.PI * 2;
  const totalMissiles = config.num_launchers * config.missiles_per_launcher;

  useFrame((state) => {
    if (radarRef.current) radarRef.current.rotation.y = state.clock.elapsedTime * 0.3;
  });

  return (
    <group position={position} quaternion={quat}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.12, 6]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} transparent opacity={0.8} />
      </mesh>
      <mesh position={[0, 0.04, 0]}>
        <boxGeometry args={[0.08, 0.08, 0.08]} />
        <meshStandardMaterial color="#374151" emissive="#1f2937" emissiveIntensity={0.3} transparent opacity={0.7} />
      </mesh>
      <mesh position={[0, 0.1, 0]}>
        <sphereGeometry args={[0.03, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.5} transparent opacity={0.6} />
      </mesh>
      <group ref={radarRef} position={[0, 0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <mesh>
          <ringGeometry args={[0, Math.min(radarRangeScaled, 5), 32, 1, 0, sectorAngle]} />
          <meshBasicMaterial color={color} transparent opacity={0.03} side={THREE.DoubleSide} />
        </mesh>
        <mesh>
          <ringGeometry args={[0, Math.min(radarRangeScaled, 5), 2, 1, 0, 0.03]} />
          <meshBasicMaterial color={color} transparent opacity={0.2} side={THREE.DoubleSide} />
        </mesh>
      </group>
      <Text position={[0, 0.2, 0]} fontSize={0.07} color={color} anchorX="center" outlineWidth={0.005} outlineColor="#000000">
        {config.name}
      </Text>
      <Text position={[0, 0.15, 0]} fontSize={0.04} color="#9ca3af" anchorX="center">
        {`${tier?.name || config.tier} · ${totalMissiles} missiles`}
      </Text>
      <Text position={[0, 0.09, 0]} fontSize={0.03} color="#6b7280" anchorX="center">
        PREVIEW
      </Text>
    </group>
  );
}

/**
 * PreviewProtectedAreaDome — Ghost protected area shown before launch
 */
export function PreviewProtectedAreaDome({ area }: { area: BuilderProtectedArea }) {
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
      material.opacity = 0.06 + Math.sin(state.clock.elapsedTime * 0.8) * 0.02;
    }
    if (ringRef.current) ringRef.current.rotation.z = state.clock.elapsedTime * 0.15;
  });

  return (
    <group position={position} quaternion={quat}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#22c55e" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 0.98, radius, 64]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.2} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[radius * 0.48, radius * 0.5, 64]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      <Text position={[0, 0.3, 0]} fontSize={0.18} color="#22c55e" anchorX="center" anchorY="middle" outlineWidth={0.01} outlineColor="#000000">
        {area.name}
      </Text>
      <Text position={[radius, 0.1, 0]} fontSize={0.08} color="#22c55e" anchorX="center">
        {`${(area.radius / 1000).toFixed(1)}km`}
      </Text>
      <Text position={[0, 0.15, 0]} fontSize={0.08} color="#4ade80" anchorX="center">
        {`P${area.priority}`}
      </Text>
    </group>
  );
}
