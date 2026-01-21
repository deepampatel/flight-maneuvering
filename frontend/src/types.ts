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
