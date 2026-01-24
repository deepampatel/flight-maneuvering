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
  // Phase 6: Physical properties
  mass?: number;
  cross_section?: number;
  drag_coefficient?: number;
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

// Phase 5: Multi-Target Support
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
  // Phase 5 additions
  intercepting_id?: string;
  intercepted_target_id?: string;
  intercepted_pairs?: [string, string][];
  assignments?: AssignmentResult;  // WTA assignments included in state
}

// Phase 5: Sensor Types
export interface SensorConfig {
  max_range: number;
  min_range: number;
  field_of_view: number;
  detection_probability: number;
  range_noise_std: number;
  angle_noise_std: number;
  update_rate: number;
}

export interface SensorDetection {
  target_id: string;
  detected: boolean;
  in_fov: boolean;
  true_range: number;
  measured_range: number;
  bearing: number;
  elevation: number;
  confidence: number;
  estimated_position?: Vec3;
}

export interface SensorDetectionsResponse {
  timestamp: number;
  detections: Record<string, SensorDetection[]>;  // interceptor_id -> detections
}

// Phase 5: WTA (Weapon-Target Assignment) Types
export interface WTAAlgorithm {
  id: string;
  name: string;
  description: string;
}

export interface Assignment {
  interceptor_id: string;
  target_id: string;
  cost: number;
  reason: string;
}

export interface AssignmentResult {
  assignments: Assignment[];
  total_cost: number;
  algorithm: string;
  unassigned_interceptors: string[];
  unassigned_targets: string[];
  timestamp: number;
}

export interface CostMatrix {
  timestamp: number;
  interceptor_ids: string[];
  target_ids: string[];
  cost_matrix: number[][];
}

// Extended Scenario type for Phase 5
export interface Scenario {
  name: string;
  description: string;
  evasion?: string;
  num_targets?: number;
  target_spacing?: number;
}

// Phase 6: Environment Types
export interface EnvironmentConfig {
  wind_velocity: Vec3;
  wind_gust_amplitude: number;
  wind_gust_period: number;
  enable_drag: boolean;
  drag_coefficient: number;
  sea_level_density: number;
}

export interface EnvironmentState {
  enabled: boolean;
  wind_velocity: Vec3;
  wind_gust_amplitude: number;
  wind_gust_period: number;
  enable_drag: boolean;
  drag_coefficient: number;
  sea_level_density: number;
  current_wind?: Vec3;
}

export interface EnvironmentConfigRequest {
  wind_speed: number;       // m/s
  wind_direction: number;   // degrees (0=North, 90=East)
  wind_gust_amplitude: number;  // m/s
  wind_gust_period: number;     // seconds
  enable_drag: boolean;
  drag_coefficient: number;
}

// Extended SimStateEvent with environment
export interface SimStateEventWithEnvironment extends SimStateEvent {
  environment?: {
    config: EnvironmentConfig;
    current_wind: Vec3;
  };
}

// Phase 6: Kalman Filter Types
export interface KalmanState {
  position: Vec3;
  velocity: Vec3;
  position_uncertainty: number;
  velocity_uncertainty: number;
  timestamp: number;
  num_updates: number;
  covariance_trace: number;
}

export interface KalmanConfig {
  process_noise_pos: number;
  process_noise_vel: number;
  measurement_noise_pos: number;
  initial_pos_variance: number;
  initial_vel_variance: number;
}

// Phase 6: Enhanced Track with Kalman
export interface SensorTrack {
  track_id: string;
  target_id: string;
  position: Vec3;
  velocity: Vec3;
  track_quality: number;
  detections: number;
  coasting: boolean;
  is_firm: boolean;
  position_uncertainty: number;
  kalman?: KalmanState;
}

export interface SensorTracksResponse {
  timestamp: number;
  tracks_by_sensor: Record<string, SensorTrack[]>;
  total_tracks: number;
}

// Phase 6: Fused Track Types
export interface FusedTrack {
  track_id: string;
  target_id: string;
  contributing_sensors: string[];
  contributing_track_ids: string[];
  position: Vec3;
  velocity: Vec3;
  position_uncertainty: number;
  confidence: number;
  last_update: number;
  num_updates: number;
}

export interface FusedTracksResponse {
  timestamp: number;
  fused_tracks: FusedTrack[];
  num_fused_tracks: number;
}

// Phase 6: Cooperative Engagement Types
export interface EngagementZone {
  zone_id: string;
  name: string;
  center: Vec3;
  dimensions: Vec3;  // width, depth, height
  rotation: number;  // heading in degrees
  assigned_interceptors: string[];
  priority: number;
  active: boolean;
  color: string;
}

export interface HandoffRequest {
  request_id: string;
  from_interceptor: string;
  to_interceptor: string;
  target_id: string;
  reason: 'fuel_low' | 'out_of_envelope' | 'reassignment' | 'zone_boundary' | 'better_geometry' | 'manual';
  status: 'pending' | 'approved' | 'executed' | 'rejected' | 'expired';
  timestamp: number;
  approved_at?: number;
  executed_at?: number;
  expiry_time: number;
}

export interface CooperativeState {
  enabled: boolean;
  engagement_zones: EngagementZone[];
  pending_handoffs: HandoffRequest[];
  completed_handoffs: HandoffRequest[];
  interceptor_zones: Record<string, string>;  // interceptor_id -> zone_id
  target_assignments: Record<string, string>;  // target_id -> interceptor_id
}

export interface EngagementZoneCreateRequest {
  name: string;
  center_x: number;
  center_y: number;
  center_z: number;
  width: number;
  depth: number;
  height: number;
  rotation?: number;
  priority?: number;
  color?: string;
}

export interface HandoffRequestCreate {
  from_interceptor: string;
  to_interceptor: string;
  target_id: string;
  reason?: string;
}

// Phase 6.4: ML/AI Types
export interface MLModelInfo {
  model_id: string;
  path: string;
  loaded: boolean;
  active: boolean;
}

export interface MLModelsResponse {
  threat_models: MLModelInfo[];
  guidance_models: MLModelInfo[];
}

export interface MLStatus {
  onnx_available: boolean;
  models: MLModelsResponse;
  active_threat_model: string | null;
  active_guidance_model: string | null;
}

export interface MLModelLoadRequest {
  model_id: string;
  model_path: string;
  model_type: 'threat_assessment' | 'guidance';
  device?: 'cpu' | 'cuda';
  num_threads?: number;
}

export interface MLModelActivateRequest {
  model_id: string;
  model_type: 'threat_assessment' | 'guidance';
}

export interface MLThreatPrediction {
  target_id: string;
  threat_score: number;
  confidence: number;
  threat_level: 'critical' | 'high' | 'medium' | 'low';
  feature_importances?: Record<string, number>;
}

export interface MLThreatAssessmentResponse {
  mode: 'ml' | 'rule' | 'hybrid';
  model_active: boolean;
  assessments: ThreatAssessment[];
}

export interface MLFeatures {
  values: number[];
  names: string[];
}

export interface MLFeaturesResponse {
  interceptor_id: string;
  target_id: string;
  threat_features: MLFeatures;
  guidance_features: MLFeatures;
}
