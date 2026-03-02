/**
 * Globe Coordinate Bridge
 *
 * Converts ENU (East-North-Up) meter positions from the simulation engine
 * into 3D Cartesian coordinates on a stylized Earth globe sphere.
 *
 * The backend physics stay in meters — this is purely a frontend visual transform.
 */

import * as THREE from 'three';

const DEG2RAD = Math.PI / 180;
const EARTH_RADIUS_M = 6_371_000;

export interface GlobeConfig {
  globeRadius: number;          // Scene units for globe sphere (default: 5)
  geoScale: number;             // Amplifies ENU spread so scenarios are visible on globe
  altitudeExaggeration: number; // Vertical exaggeration for flight altitude
  originLat: number;            // Scenario geographic center (degrees)
  originLng: number;
}

export const DEFAULT_GLOBE_CONFIG: GlobeConfig = {
  globeRadius: 5,
  geoScale: 300,
  altitudeExaggeration: 50,
  originLat: 31.5,
  originLng: 34.5,
};

/**
 * Convert ENU meter position to lat/lng (radians) relative to scenario origin.
 * x = East, y = North, z = Up.
 */
function enuToLatLng(
  enu: { x: number; y: number; z: number },
  originLatRad: number,
  originLngRad: number,
  geoScale: number,
): { lat: number; lng: number; alt: number } {
  const scaledEast = enu.x * geoScale;
  const scaledNorth = enu.y * geoScale;

  const dLat = scaledNorth / EARTH_RADIUS_M;
  const dLng = scaledEast / (EARTH_RADIUS_M * Math.cos(originLatRad));

  return {
    lat: originLatRad + dLat,
    lng: originLngRad + dLng,
    alt: enu.z,
  };
}

/**
 * Convert spherical (lat, lng in radians, radius) to Three.js Cartesian.
 * Y-up convention matching Three.js default.
 */
function sphericalToCartesian(
  lat: number,
  lng: number,
  r: number,
): [number, number, number] {
  return [
    r * Math.cos(lat) * Math.sin(lng),
    r * Math.sin(lat),
    r * Math.cos(lat) * Math.cos(lng),
  ];
}

/**
 * Main converter: ENU meters → Three.js globe position [x, y, z].
 */
export function metersToGlobe(
  enu: { x: number; y: number; z: number },
  config: GlobeConfig,
): [number, number, number] {
  const originLatRad = config.originLat * DEG2RAD;
  const originLngRad = config.originLng * DEG2RAD;

  const { lat, lng, alt } = enuToLatLng(enu, originLatRad, originLngRad, config.geoScale);

  const altScene = alt * config.altitudeExaggeration * (config.globeRadius / EARTH_RADIUS_M);
  const r = config.globeRadius + altScene;

  return sphericalToCartesian(lat, lng, r);
}

/**
 * Surface normal at a given ENU position on the globe (points outward).
 * Used to orient ground markers perpendicular to the globe surface.
 */
export function globeSurfaceNormal(
  enu: { x: number; y: number; z: number },
  config: GlobeConfig,
): THREE.Vector3 {
  const pos = metersToGlobe({ x: enu.x, y: enu.y, z: 0 }, config);
  return new THREE.Vector3(pos[0], pos[1], pos[2]).normalize();
}

/**
 * Transform ENU velocity vector to globe-frame direction.
 *
 * At a point on the globe, the local tangent frame is:
 *   east  = d/dLng direction (tangent to surface, pointing east)
 *   north = d/dLat direction (tangent to surface, pointing north)
 *   up    = surface normal (radially outward)
 *
 * The ENU velocity (vx=East, vy=North, vz=Up) is expressed in this basis.
 */
export function enuVelocityToGlobe(
  enuPos: { x: number; y: number; z: number },
  enuVel: { x: number; y: number; z: number },
  config: GlobeConfig,
): THREE.Vector3 {
  const originLatRad = config.originLat * DEG2RAD;
  const originLngRad = config.originLng * DEG2RAD;
  const { lat, lng } = enuToLatLng(enuPos, originLatRad, originLngRad, config.geoScale);

  // Local tangent basis vectors in Three.js Cartesian (Y-up)
  // East = d/dLng of sphericalToCartesian
  const east = new THREE.Vector3(
    Math.cos(lat) * Math.cos(lng),
    0,
    -Math.cos(lat) * Math.sin(lng),
  ).normalize();

  // North = d/dLat of sphericalToCartesian
  const north = new THREE.Vector3(
    -Math.sin(lat) * Math.sin(lng),
    Math.cos(lat),
    -Math.sin(lat) * Math.cos(lng),
  ).normalize();

  // Up = surface normal (radially outward)
  const up = new THREE.Vector3(
    Math.cos(lat) * Math.sin(lng),
    Math.sin(lat),
    Math.cos(lat) * Math.cos(lng),
  ).normalize();

  // Transform ENU velocity to globe frame
  const result = new THREE.Vector3();
  result.addScaledVector(east, enuVel.x);
  result.addScaledVector(north, enuVel.y);
  result.addScaledVector(up, enuVel.z);

  return result;
}

/**
 * Quaternion to orient a group so its local Y-up aligns with the globe surface normal.
 */
const _upVec = new THREE.Vector3(0, 1, 0);
export function surfaceQuaternion(
  enu: { x: number; y: number; z: number },
  config: GlobeConfig,
): THREE.Quaternion {
  const normal = globeSurfaceNormal(enu, config);
  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(_upVec, normal);
  return quat;
}

/**
 * Convert a distance in meters to approximate scene units on the globe surface.
 * Useful for radii (radar range, protected area radius) that need consistent scaling.
 */
export function metersToGlobeLength(meters: number, config: GlobeConfig): number {
  return meters * config.geoScale * config.globeRadius / EARTH_RADIUS_M;
}

// ---------------------------------------------------------------------------
// Unified scene transforms — dispatch by viewMode ('sim' | 'globe')
// ---------------------------------------------------------------------------

/** Scale factor for simulation (flat) view: 1 unit = 1 km */
export const SIM_SCALE = 0.001;

/** ENU meters → scene-space position [x, y, z]. */
export function metersToScene(
  enu: { x: number; y: number; z: number },
  viewMode: 'sim' | 'globe',
  config: GlobeConfig,
): [number, number, number] {
  if (viewMode === 'sim') {
    // ENU → Three.js Y-up: x=East, y=Up, z=-North
    return [
      enu.x * SIM_SCALE,
      enu.z * SIM_SCALE,
      -enu.y * SIM_SCALE,
    ];
  }
  return metersToGlobe(enu, config);
}

/** ENU velocity → scene-frame direction vector. */
export function velocityToScene(
  enuPos: { x: number; y: number; z: number },
  enuVel: { x: number; y: number; z: number },
  viewMode: 'sim' | 'globe',
  config: GlobeConfig,
): THREE.Vector3 {
  if (viewMode === 'sim') {
    return new THREE.Vector3(
      enuVel.x * SIM_SCALE,
      enuVel.z * SIM_SCALE,
      -enuVel.y * SIM_SCALE,
    );
  }
  return enuVelocityToGlobe(enuPos, enuVel, config);
}

/** Quaternion to orient an object perpendicular to the surface. */
export function sceneSurfaceQuaternion(
  enu: { x: number; y: number; z: number },
  viewMode: 'sim' | 'globe',
  config: GlobeConfig,
): THREE.Quaternion {
  if (viewMode === 'sim') {
    return new THREE.Quaternion(); // identity — flat ground, Y already up
  }
  return surfaceQuaternion(enu, config);
}

/** Convert a distance in meters to scene units. */
export function metersToSceneLength(
  meters: number,
  viewMode: 'sim' | 'globe',
  config: GlobeConfig,
): number {
  if (viewMode === 'sim') return meters * SIM_SCALE;
  return metersToGlobeLength(meters, config);
}

/** Surface normal at an ENU position. */
export function sceneSurfaceNormal(
  enu: { x: number; y: number; z: number },
  viewMode: 'sim' | 'globe',
  config: GlobeConfig,
): THREE.Vector3 {
  if (viewMode === 'sim') return new THREE.Vector3(0, 1, 0);
  return globeSurfaceNormal(enu, config);
}

/** Offset a scene position away from the surface by `offset` units. */
export function offsetFromSurface(
  pos: [number, number, number],
  offset: number,
  viewMode: 'sim' | 'globe',
): [number, number, number] {
  if (viewMode === 'sim') {
    return [pos[0], pos[1] + offset, pos[2]];
  }
  // Globe: offset radially outward
  const len = Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
  if (len < 0.001) return [pos[0], pos[1] + offset, pos[2]];
  const factor = offset / len;
  return [pos[0] * (1 + factor), pos[1] * (1 + factor), pos[2] * (1 + factor)];
}

// ---------------------------------------------------------------------------
// Google Maps helpers — real geographic coordinates (geoScale = 1)
// ---------------------------------------------------------------------------

/**
 * Convert ENU meter position to real lat/lng degrees for Google Maps.
 * Uses geoScale=1 (no amplification) — gives actual geographic coordinates.
 *
 * Unlike metersToGlobe() which amplifies positions 300× for visibility on a
 * small 3D sphere, this returns where entities actually are on Earth.
 */
export function enuToRealLatLng(
  enu: { x: number; y: number; z: number },
  originLatDeg: number,
  originLngDeg: number,
): { lat: number; lng: number; alt: number } {
  const originLatRad = originLatDeg * DEG2RAD;
  const originLngRad = originLngDeg * DEG2RAD;
  const result = enuToLatLng(enu, originLatRad, originLngRad, 1); // geoScale = 1
  return {
    lat: result.lat / DEG2RAD,  // radians → degrees
    lng: result.lng / DEG2RAD,
    alt: enu.z,
  };
}
