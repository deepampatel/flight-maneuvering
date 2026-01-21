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

// Phase 4: Intercept Geometry types
export interface InterceptGeometry {
  interceptor_id: string;
  target_id: string;
  los_range: number;              // Distance to target (meters)
  los_rate_magnitude: number;     // Angular rate of LOS (rad/s)
  aspect_angle: number;           // 0=head-on, 180=tail (degrees)
  antenna_train_angle: number;    // Angle off interceptor nose (degrees)
  lead_angle: number;             // Required lead for collision (degrees)
  collision_course: boolean;      // True if current heading leads to intercept
  time_to_intercept: number;      // Estimated TTI (seconds), -1 if not closing
  predicted_miss_distance: number; // Miss at CPA if no maneuver (meters)
  closing_velocity: number;       // Rate of range decrease (m/s)
  intercept_point?: Vec3;         // Where collision will occur
}

export interface InterceptGeometryResponse {
  timestamp: number;
  geometries: InterceptGeometry[];
}

// Phase 4: Threat Assessment types
export interface ThreatScore {
  target_id: string;
  total_score: number;            // 0-100
  threat_level: 'critical' | 'high' | 'medium' | 'low';
  time_score: number;             // Component scores (0-1)
  closing_score: number;
  aspect_score: number;
  altitude_score: number;
  maneuver_score: number;
  time_to_impact: number;         // seconds
  closing_velocity: number;       // m/s
  aspect_angle: number;           // degrees
  altitude_delta: number;         // meters
  priority_rank: number;          // 1 = highest priority
}

export interface ThreatAssessment {
  timestamp: number;
  assessor_id: string;            // Which interceptor is assessing
  threats: ThreatScore[];
  highest_threat_id: string;
  engagement_recommendation: 'engage' | 'monitor' | 'ignore';
}

export interface ThreatAssessmentResponse {
  assessments: ThreatAssessment[];
}

// Phase 4: Recording & Replay types
export interface RecordingMetadata {
  recording_id: string;
  created_at: number;
  scenario_name: string;
  result: string;
  final_miss_distance: number;
  total_sim_time: number;
  total_ticks: number;
  guidance: string;
  evasion: string;
}

export interface RecordingListResponse {
  recordings: RecordingMetadata[];
  total: number;
}

export interface ReplayConfig {
  speed_multiplier: number;
  start_tick: number;
  end_tick?: number;
}

export interface ReplayState {
  recording_id: string;
  is_playing: boolean;
  is_paused: boolean;
  current_tick: number;
  total_ticks: number;
  speed_multiplier: number;
  scenario_name: string;
  result: string;
}
