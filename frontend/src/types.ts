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
