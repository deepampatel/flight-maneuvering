/**
 * Defense tier presets and threat type catalog.
 * Mirrors the backend /api/catalog response for offline use.
 */

export interface TierDefaults {
  name: string;
  interceptor: string;
  color: string;
  minRange: number;
  maxRange: number;
  defaults: {
    radar_range: number;
    radar_sector: number;
    num_launchers: number;
    missiles_per_launcher: number;
    max_simultaneous: number;
    launch_speed: number;
    launch_elevation: number;
    min_altitude: number;
  };
}

export interface ThreatTypeInfo {
  name: string;
  speed: number;
  altitude: number;
  category: string;
}

export const TIER_DEFAULTS: Record<string, TierDefaults> = {
  iron_dome: {
    name: 'Iron Dome',
    interceptor: 'tamir',
    color: '#3b82f6',
    minRange: 4000,
    maxRange: 70000,
    defaults: {
      radar_range: 70000,
      radar_sector: 360,
      num_launchers: 3,
      missiles_per_launcher: 20,
      max_simultaneous: 6,
      launch_speed: 250,
      launch_elevation: 80,
      min_altitude: 100,
    },
  },
  davids_sling: {
    name: "David's Sling",
    interceptor: 'stunner',
    color: '#06b6d4',
    minRange: 40000,
    maxRange: 300000,
    defaults: {
      radar_range: 300000,
      radar_sector: 360,
      num_launchers: 4,
      missiles_per_launcher: 12,
      max_simultaneous: 4,
      launch_speed: 800,
      launch_elevation: 85,
      min_altitude: 500,
    },
  },
  arrow: {
    name: 'Arrow',
    interceptor: 'arrow_3',
    color: '#8b5cf6',
    minRange: 100000,
    maxRange: 2400000,
    defaults: {
      radar_range: 2400000,
      radar_sector: 120,
      num_launchers: 6,
      missiles_per_launcher: 4,
      max_simultaneous: 2,
      launch_speed: 2500,
      launch_elevation: 89,
      min_altitude: 10000,
    },
  },
};

export const THREAT_TYPES: Record<string, ThreatTypeInfo> = {
  qassam: { name: 'Qassam', speed: 200, altitude: 2000, category: 'short_range' },
  grad: { name: 'Grad', speed: 300, altitude: 3000, category: 'medium_range' },
  cruise_missile: { name: 'Cruise Missile', speed: 250, altitude: 500, category: 'guided' },
};

export const TIER_IDS = ['iron_dome', 'davids_sling', 'arrow'] as const;
export type TierId = typeof TIER_IDS[number];
