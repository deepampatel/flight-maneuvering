/**
 * Shared Types - The Contract Between Frontend and Backend
 *
 * These types mirror the backend's data structures.
 * In a real system, you'd generate these from a schema (OpenAPI, protobuf).
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface EntityState {
  id: string;
  type: 'target' | 'interceptor';
  position: Vec3;
  velocity: Vec3;
  acceleration: Vec3;
  speed: number;
}

export interface SimStateEvent {
  type: 'state';
  run_id: string;
  ts: number;
  sim_time: number;
  tick: number;
  status: 'ready' | 'running' | 'paused' | 'completed';
  result: 'pending' | 'intercept' | 'timeout' | 'missed';
  entities: EntityState[];
  miss_distance: number;
}

export interface SimCompleteEvent {
  type: 'complete';
  run_id: string;
  ts: number;
  result: 'intercept' | 'timeout' | 'missed';
  final_miss_distance: number;
  sim_time: number;
  ticks: number;
}

export type SimEvent = SimStateEvent | SimCompleteEvent;

export interface Scenario {
  name: string;
  description: string;
  evasion?: string;
}

export interface EvasionType {
  id: string;
  name: string;
  description: string;
}

export interface GuidanceLaw {
  id: string;
  name: string;
  description: string;
}

// Monte Carlo types
export interface MonteCarloConfig {
  scenario: string;
  guidance: string;
  nav_constant: number;
  num_runs: number;
  kill_radius: number;
  position_noise_std: number;
  velocity_noise_std: number;
}

export interface MonteCarloResults {
  config: Record<string, unknown>;
  num_runs: number;
  intercept_rate: number;
  miss_rate: number;
  timeout_rate: number;
  mean_miss_distance: number;
  std_miss_distance: number;
  min_miss_distance: number;
  max_miss_distance: number;
  mean_time_to_intercept: number;
  miss_distance_histogram: {
    bin_edges: number[];
    counts: number[];
  };
}

export interface ParameterSweepResults {
  param_name: string;
  results: MonteCarloResults[];
}

// Engagement Envelope types
export interface EnvelopeConfig {
  range_min: number;
  range_max: number;
  range_steps: number;
  bearing_min: number;
  bearing_max: number;
  bearing_steps: number;
  elevation_min: number;
  elevation_max: number;
  elevation_steps: number;
  runs_per_point: number;
  guidance: string;
  nav_constant: number;
  kill_radius: number;
  target_speed: number;
  evasion: string;
  interceptor_speed: number;
}

export interface EnvelopePoint {
  range: number;
  bearing: number;
  elevation: number;
  intercept_rate: number;
  mean_miss_distance: number;
  mean_time_to_intercept: number;
}

export interface Heatmap2D {
  data: number[][];
  x_label: string;
  y_label: string;
  x_values: number[];
  y_values: number[];
}

export interface Surface3DVertex {
  x: number;
  y: number;
  z: number;
  range: number;
  bearing: number;
  elevation: number;
  intercept_rate: number;
}

export interface Surface3D {
  surfaces: {
    elevation: number;
    vertices: Surface3DVertex[];
  }[];
}

export interface EnvelopeResults {
  config: Record<string, unknown>;
  range_values: number[];
  bearing_values: number[];
  elevation_values: number[];
  heatmap_2d: Heatmap2D;
  surface_3d: Surface3D;
  points: EnvelopePoint[];
}
