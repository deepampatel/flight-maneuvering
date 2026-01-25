/**
 * useSimulation Hook - WebSocket Connection Management
 *
 * This hook manages the real-time connection to the simulation server.
 *
 * Key concepts:
 * 1. WebSocket for low-latency streaming (vs HTTP polling)
 * 2. Auto-reconnect on disconnect
 * 3. State management for UI updates
 *
 * The hook provides:
 * - Connection status
 * - Latest simulation state
 * - Methods to start/stop runs
 * - Monte Carlo analysis
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  SimStateEvent,
  SimEvent,
  Scenario,
  GuidanceLaw,
  EvasionType,
  MonteCarloResults,
  ParameterSweepResults,
  EnvelopeConfig,
  EnvelopeResults,
  InterceptGeometry,
  ThreatAssessment,
  RecordingMetadata,
  ReplayState,
  ReplayConfig,
  //
  SensorDetection,
  AssignmentResult,
  WTAAlgorithm,
  CostMatrix,
  //
  EnvironmentState,
  SensorTrack,
  SensorTracksResponse,
  FusedTrack,
  FusedTracksResponse,
  CooperativeState,
  EngagementZoneCreateRequest,
  HandoffRequestCreate,
  // ML
  MLStatus,
  //
  SwarmStatus,
  SwarmConfig,
  FormationInfo,
  FormationType,
  HMTStatus,
  HMTConfig,
  PendingAction,
  AuthorityLevel,
  AuthorityLevelInfo,
  DatalinkStatus,
  TerrainStatus,
  Phase7Status,
  // Launchers
  LauncherState,
} from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

// Generic fetch helper to reduce repetitive code
async function fetchApi<T>(
  endpoint: string,
  setter: (data: T) => void,
  extractor?: (data: unknown) => T
): Promise<void> {
  try {
    const response = await fetch(`${API_URL}${endpoint}`);
    if (response.ok) {
      const data = await response.json();
      setter(extractor ? extractor(data) : data);
    }
  } catch {
    // Silently fail if no active simulation
  }
}

// Generic POST helper
async function postApi(
  endpoint: string,
  body?: unknown,
  onSuccess?: () => Promise<void>
): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (response.ok && onSuccess) {
      await onSuccess();
    }
    return response.ok;
  } catch {
    return false;
  }
}

interface PlannedEntity {
  id: string;
  type: string;
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
}

interface PlannedZone {
  id: string;
  name: string;
  center: { x: number; y: number; z: number };
  dimensions: { x: number; y: number; z: number };
  color: string;
}

interface PlannedLauncher {
  id: string;
  position: { x: number; y: number; z: number };
  detection_range: number;
  num_missiles: number;
  launch_mode: string;
}

interface RunOptions {
  scenario: string;
  guidance: string;
  navConstant: number;
  evasion?: string;
  numInterceptors?: number;
  numTargets?: number;  // Multi-target support
  wtaAlgorithm?: string;  // WTA algorithm selection
  // Environment
  windSpeed?: number;
  windDirection?: number;
  windGusts?: number;
  enableDrag?: boolean;
  // Cooperative
  enableCooperative?: boolean;
  // Mission Planner: Custom entities
  customEntities?: PlannedEntity[];
  customZones?: PlannedZone[];
  customLaunchers?: PlannedLauncher[];
  // Swarm
  enableSwarm?: boolean;
  swarmFormation?: FormationType;
  swarmSpacing?: number;
  // HMT
  enableHmt?: boolean;
  hmtAuthorityLevel?: AuthorityLevel;
  // Datalink
  enableDatalink?: boolean;
  // Terrain
  enableTerrain?: boolean;
}

interface MonteCarloOptions {
  scenario: string;
  guidance: string;
  navConstant: number;
  numRuns: number;
  killRadius: number;
  positionNoiseStd: number;
  velocityNoiseStd: number;
}

interface UseSimulationReturn {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  guidanceLaws: GuidanceLaw[];
  evasionTypes: EvasionType[];
  startRun: (options: RunOptions) => Promise<void>;
  stopRun: () => Promise<void>;
  runMonteCarlo: (options: MonteCarloOptions) => Promise<MonteCarloResults>;
  runParameterSweep: (
    scenario: string,
    guidance: string,
    paramName: string,
    paramValues: number[],
    numRunsPerValue: number
  ) => Promise<ParameterSweepResults>;
  runEnvelopeAnalysis: (config: Partial<EnvelopeConfig>) => Promise<EnvelopeResults>;
  monteCarloLoading: boolean;
  envelopeLoading: boolean;
  // Intercept Geometry
  interceptGeometry: InterceptGeometry[] | null;
  fetchInterceptGeometry: () => Promise<void>;
  // Threat Assessment
  threatAssessment: ThreatAssessment[] | null;
  fetchThreatAssessment: () => Promise<void>;
  // Recording
  isRecording: boolean;
  recordings: RecordingMetadata[];
  startRecording: () => Promise<string>;
  stopRecording: () => Promise<void>;
  refreshRecordings: () => Promise<void>;
  deleteRecording: (recordingId: string) => Promise<void>;
  // Replay
  replayState: ReplayState | null;
  startReplay: (recordingId: string, config?: Partial<ReplayConfig>) => Promise<void>;
  pauseReplay: () => Promise<void>;
  resumeReplay: () => Promise<void>;
  seekReplay: (tick: number) => Promise<void>;
  stopReplay: () => Promise<void>;
  // Sensor Detections
  sensorDetections: Record<string, SensorDetection[]> | null;
  fetchSensorDetections: () => Promise<void>;
  // WTA
  wtaAlgorithms: WTAAlgorithm[];
  assignments: AssignmentResult | null;
  fetchAssignments: (algorithm?: string) => Promise<void>;
  costMatrix: CostMatrix | null;
  fetchCostMatrix: () => Promise<void>;
  // Environment
  environmentState: EnvironmentState | null;
  fetchEnvironmentState: () => Promise<void>;
  // Kalman & Fusion
  sensorTracks: SensorTrack[] | null;
  fusedTracks: FusedTrack[] | null;
  fetchSensorTracks: () => Promise<void>;
  fetchFusedTracks: () => Promise<void>;
  // Cooperative Engagement
  cooperativeState: CooperativeState | null;
  fetchCooperativeState: () => Promise<void>;
  createEngagementZone: (zone: EngagementZoneCreateRequest) => Promise<void>;
  deleteEngagementZone: (zoneId: string) => Promise<void>;
  assignInterceptorToZone: (interceptorId: string, zoneId: string) => Promise<void>;
  requestHandoff: (request: HandoffRequestCreate) => Promise<void>;
  // ML
  mlStatus: MLStatus | null;
  fetchMLStatus: () => Promise<void>;
  // Swarm
  swarmStatus: SwarmStatus | null;
  formationTypes: FormationInfo[];
  fetchSwarmStatus: () => Promise<void>;
  configureSwarm: (config: Partial<SwarmConfig>) => Promise<void>;
  setSwarmFormation: (formation: FormationType) => Promise<void>;
  // HMT
  hmtStatus: HMTStatus | null;
  authorityLevels: AuthorityLevelInfo[];
  pendingActions: PendingAction[];
  fetchHMTStatus: () => Promise<void>;
  fetchPendingActions: () => Promise<void>;
  approveAction: (actionId: string, reason?: string) => Promise<void>;
  rejectAction: (actionId: string, reason?: string) => Promise<void>;
  setAuthorityLevel: (level: AuthorityLevel) => Promise<void>;
  configureHMT: (config: Partial<HMTConfig>) => Promise<void>;
  // Datalink
  datalinkStatus: DatalinkStatus | null;
  fetchDatalinkStatus: () => Promise<void>;
  // Terrain
  terrainStatus: TerrainStatus | null;
  fetchTerrainStatus: () => Promise<void>;
  // Combined status
  phase7Status: Phase7Status | null;
  fetchPhase7Status: () => Promise<void>;
  // Launchers
  launchers: LauncherState[] | null;
}

export function useSimulation(): UseSimulationReturn {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SimStateEvent | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, Scenario>>({});
  const [guidanceLaws, setGuidanceLaws] = useState<GuidanceLaw[]>([]);
  const [evasionTypes, setEvasionTypes] = useState<EvasionType[]>([]);
  const [monteCarloLoading, setMonteCarloLoading] = useState(false);
  const [envelopeLoading, setEnvelopeLoading] = useState(false);
  // Intercept Geometry & Threat Assessment
  const [interceptGeometry, setInterceptGeometry] = useState<InterceptGeometry[] | null>(null);
  const [threatAssessment, setThreatAssessment] = useState<ThreatAssessment[] | null>(null);
  // Recording & Replay
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState<RecordingMetadata[]>([]);
  const [replayState, setReplayState] = useState<ReplayState | null>(null);
  // Sensors & WTA
  const [sensorDetections, setSensorDetections] = useState<Record<string, SensorDetection[]> | null>(null);
  const [wtaAlgorithms, setWtaAlgorithms] = useState<WTAAlgorithm[]>([]);
  const [assignments, setAssignments] = useState<AssignmentResult | null>(null);
  const [costMatrix, setCostMatrix] = useState<CostMatrix | null>(null);
  // Environment
  const [environmentState, setEnvironmentState] = useState<EnvironmentState | null>(null);
  // Kalman & Fusion
  const [sensorTracks, setSensorTracks] = useState<SensorTrack[] | null>(null);
  const [fusedTracks, setFusedTracks] = useState<FusedTrack[] | null>(null);
  // Cooperative Engagement
  const [cooperativeState, setCooperativeState] = useState<CooperativeState | null>(null);
  // ML
  const [mlStatus, setMLStatus] = useState<MLStatus | null>(null);
  // Swarm
  const [swarmStatus, setSwarmStatus] = useState<SwarmStatus | null>(null);
  const [formationTypes, setFormationTypes] = useState<FormationInfo[]>([]);
  // HMT
  const [hmtStatus, setHMTStatus] = useState<HMTStatus | null>(null);
  const [authorityLevels, setAuthorityLevels] = useState<AuthorityLevelInfo[]>([]);
  const [pendingActions, setPendingActions] = useState<PendingAction[]>([]);
  // Datalink
  const [datalinkStatus, setDatalinkStatus] = useState<DatalinkStatus | null>(null);
  // Terrain
  const [terrainStatus, setTerrainStatus] = useState<TerrainStatus | null>(null);
  // Combined
  const [phase7Status, setPhase7Status] = useState<Phase7Status | null>(null);
  // Launchers
  const [launchers, setLaunchers] = useState<LauncherState[] | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);

  // Fetch available scenarios, guidance laws, and evasion types on mount
  useEffect(() => {
    fetch(`${API_URL}/scenarios`)
      .then((res) => res.json())
      .then(setScenarios)
      .catch(console.error);

    fetch(`${API_URL}/guidance`)
      .then((res) => res.json())
      .then((data) => setGuidanceLaws(data.guidance_laws))
      .catch(console.error);

    fetch(`${API_URL}/evasion`)
      .then((res) => res.json())
      .then((data) => setEvasionTypes(data.evasion_types))
      .catch(console.error);

    // Fetch WTA algorithms
    fetch(`${API_URL}/wta/algorithms`)
      .then((res) => res.json())
      .then((data) => setWtaAlgorithms(data.algorithms))
      .catch(console.error);

    // Fetch formation types
    fetch(`${API_URL}/swarm/formations`)
      .then((res) => res.json())
      .then((data) => setFormationTypes(data.formations || []))
      .catch(console.error);

    // Fetch authority levels
    fetch(`${API_URL}/hmt/authority-levels`)
      .then((res) => res.json())
      .then((data) => setAuthorityLevels(data.authority_levels || []))
      .catch(console.error);

    // Fetch combined status
    fetch(`${API_URL}/phase7/status`)
      .then((res) => res.json())
      .then((data) => setPhase7Status(data))
      .catch(console.error);
  }, []);

  // WebSocket connection management
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: SimEvent = JSON.parse(event.data);

        if (data.type === 'state') {
          setState(data);
          // Extract assignments from state event if present
          if (data.assignments) {
            setAssignments(data.assignments);
          }
          // Extract HMT pending actions from state event
          if (data.hmt?.pending_actions) {
            setPendingActions(data.hmt.pending_actions);
          }
          // Extract launchers from state event
          if (data.launchers) {
            setLaunchers(data.launchers);
          }
        } else if (data.type === 'complete') {
          console.log('Simulation complete:', data.result);
        }
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      wsRef.current = null;

      // Auto-reconnect after 2 seconds
      reconnectTimeoutRef.current = window.setTimeout(() => {
        console.log('Attempting reconnect...');
        connect();
      }, 2000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, []);

  // Connect on mount, cleanup on unmount
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Start a new simulation run
  const startRun = useCallback(async (options: RunOptions) => {
    setState(null); // Clear previous state
    // Clear sensor/WTA state
    setSensorDetections(null);
    setAssignments(null);
    setCostMatrix(null);

    const response = await fetch(`${API_URL}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: options.scenario,
        guidance: options.guidance,
        nav_constant: options.navConstant,
        evasion: options.evasion || 'none',
        num_interceptors: options.numInterceptors || 1,
        num_targets: options.numTargets,  // Multi-target
        wta_algorithm: options.wtaAlgorithm || 'hungarian',  // WTA algorithm
        real_time: true,
        // Environment
        wind_speed: options.windSpeed || 0,
        wind_direction: options.windDirection || 0,
        wind_gusts: options.windGusts || 0,
        enable_drag: options.enableDrag || false,
        // Cooperative
        enable_cooperative: options.enableCooperative || false,
        // Mission Planner: Custom entities
        custom_entities: options.customEntities,
        custom_zones: options.customZones,
        custom_launchers: options.customLaunchers,
        // Swarm
        enable_swarm: options.enableSwarm || false,
        swarm_formation: options.swarmFormation,
        swarm_spacing: options.swarmSpacing,
        // HMT
        enable_hmt: options.enableHmt || false,
        hmt_authority_level: options.hmtAuthorityLevel,
        // Datalink
        enable_datalink: options.enableDatalink || false,
        // Terrain
        enable_terrain: options.enableTerrain || false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start run: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Run started:', data);
  }, []);

  // Stop current run
  const stopRun = useCallback(async () => {
    await fetch(`${API_URL}/runs/stop`, { method: 'POST' });
  }, []);

  // Run Monte Carlo analysis
  const runMonteCarlo = useCallback(
    async (options: MonteCarloOptions): Promise<MonteCarloResults> => {
      setMonteCarloLoading(true);
      try {
        const response = await fetch(`${API_URL}/monte-carlo`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scenario: options.scenario,
            guidance: options.guidance,
            nav_constant: options.navConstant,
            num_runs: options.numRuns,
            kill_radius: options.killRadius,
            position_noise_std: options.positionNoiseStd,
            velocity_noise_std: options.velocityNoiseStd,
          }),
        });

        if (!response.ok) {
          throw new Error(`Monte Carlo failed: ${response.statusText}`);
        }

        return await response.json();
      } finally {
        setMonteCarloLoading(false);
      }
    },
    []
  );

  // Run parameter sweep
  const runParameterSweep = useCallback(
    async (
      scenario: string,
      guidance: string,
      paramName: string,
      paramValues: number[],
      numRunsPerValue: number
    ): Promise<ParameterSweepResults> => {
      setMonteCarloLoading(true);
      try {
        const response = await fetch(`${API_URL}/monte-carlo/sweep`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scenario,
            guidance,
            param_name: paramName,
            param_values: paramValues,
            num_runs_per_value: numRunsPerValue,
          }),
        });

        if (!response.ok) {
          throw new Error(`Parameter sweep failed: ${response.statusText}`);
        }

        return await response.json();
      } finally {
        setMonteCarloLoading(false);
      }
    },
    []
  );

  // Run engagement envelope analysis
  const runEnvelopeAnalysis = useCallback(
    async (config: Partial<EnvelopeConfig>): Promise<EnvelopeResults> => {
      setEnvelopeLoading(true);
      try {
        const response = await fetch(`${API_URL}/envelope`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        });

        if (!response.ok) {
          throw new Error(`Envelope analysis failed: ${response.statusText}`);
        }

        return await response.json();
      } finally {
        setEnvelopeLoading(false);
      }
    },
    []
  );

  const fetchInterceptGeometry = useCallback(
    () => fetchApi('/intercept-geometry', setInterceptGeometry, (d: any) => d.geometries), []
  );

  const fetchThreatAssessment = useCallback(
    () => fetchApi('/threat-assessment', setThreatAssessment, (d: any) => d.assessments), []
  );

  // Start recording
  const startRecording = useCallback(async (): Promise<string> => {
    const response = await fetch(`${API_URL}/recordings/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await response.json();
    if (data.status === 'recording_started' || data.status === 'already_recording') {
      setIsRecording(true);
    }
    return data.recording_id;
  }, []);

  // Stop recording
  const stopRecording = useCallback(async () => {
    const response = await fetch(`${API_URL}/recordings/stop`, {
      method: 'POST',
    });
    const data = await response.json();
    if (data.status === 'recording_saved') {
      setIsRecording(false);
      // Refresh recordings list
      const listResponse = await fetch(`${API_URL}/recordings`);
      const listData = await listResponse.json();
      setRecordings(listData.recordings);
    }
  }, []);

  // Refresh recordings list
  const refreshRecordings = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/recordings`);
      const data = await response.json();
      setRecordings(data.recordings);
    } catch (e) {
      console.error('Failed to fetch recordings:', e);
    }
  }, []);

  // Delete recording
  const deleteRecording = useCallback(async (recordingId: string) => {
    await fetch(`${API_URL}/recordings/${recordingId}`, {
      method: 'DELETE',
    });
    // Refresh list
    const response = await fetch(`${API_URL}/recordings`);
    const data = await response.json();
    setRecordings(data.recordings);
  }, []);

  // Start replay
  const startReplay = useCallback(async (recordingId: string, config?: Partial<ReplayConfig>) => {
    setState(null); // Clear previous state
    const response = await fetch(`${API_URL}/replay/${recordingId}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config || {}),
    });
    const data = await response.json();
    if (data.status === 'replay_started') {
      // Fetch replay state
      const stateResponse = await fetch(`${API_URL}/replay/state`);
      const stateData = await stateResponse.json();
      if (stateData.recording_id) {
        setReplayState(stateData);
      }
    }
  }, []);

  // Pause replay
  const pauseReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/pause`, { method: 'POST' });
    // Update replay state
    const response = await fetch(`${API_URL}/replay/state`);
    const data = await response.json();
    if (data.recording_id) {
      setReplayState(data);
    }
  }, []);

  // Resume replay
  const resumeReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/resume`, { method: 'POST' });
    const response = await fetch(`${API_URL}/replay/state`);
    const data = await response.json();
    if (data.recording_id) {
      setReplayState(data);
    }
  }, []);

  // Seek replay
  const seekReplay = useCallback(async (tick: number) => {
    await fetch(`${API_URL}/replay/seek?tick=${tick}`, { method: 'POST' });
  }, []);

  // Stop replay
  const stopReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/stop`, { method: 'POST' });
    setReplayState(null);
  }, []);

  // Fetch recordings on mount
  useEffect(() => {
    refreshRecordings();
  }, [refreshRecordings]);

  const fetchSensorDetections = useCallback(
    () => fetchApi('/sensor/detections', setSensorDetections, (d: any) => d.detections), []
  );

  // Fetch WTA assignments
  const fetchAssignments = useCallback(async (algorithm: string = 'greedy_nearest') => {
    try {
      const response = await fetch(`${API_URL}/wta/assignments?algorithm=${algorithm}`);
      if (response.ok) {
        const data = await response.json();
        setAssignments(data);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  const fetchCostMatrix = useCallback(
    () => fetchApi('/wta/cost-matrix', setCostMatrix), []
  );

  const fetchEnvironmentState = useCallback(
    () => fetchApi('/environment/config', setEnvironmentState), []
  );

  const fetchSensorTracks = useCallback(
    () => fetchApi('/sensor/tracks', setSensorTracks, (d: SensorTracksResponse) => {
      const allTracks: SensorTrack[] = [];
      for (const tracks of Object.values(d.tracks_by_sensor)) {
        allTracks.push(...tracks);
      }
      return allTracks;
    }), []
  );

  const fetchFusedTracks = useCallback(
    () => fetchApi('/sensor/fused-tracks', setFusedTracks, (d: FusedTracksResponse) => d.fused_tracks), []
  );

  const fetchCooperativeState = useCallback(
    () => fetchApi('/cooperative/state', setCooperativeState), []
  );

  const createEngagementZone = useCallback(
    (zone: EngagementZoneCreateRequest) => postApi('/cooperative/zones', zone, fetchCooperativeState),
    [fetchCooperativeState]
  );

  const deleteEngagementZone = useCallback(async (zoneId: string) => {
    try {
      const response = await fetch(`${API_URL}/cooperative/zones/${zoneId}`, { method: 'DELETE' });
      if (response.ok) await fetchCooperativeState();
    } catch { /* ignore */ }
  }, [fetchCooperativeState]);

  const assignInterceptorToZone = useCallback(
    (interceptorId: string, zoneId: string) =>
      postApi('/cooperative/zones/assign', { interceptor_id: interceptorId, zone_id: zoneId }, fetchCooperativeState),
    [fetchCooperativeState]
  );

  const requestHandoff = useCallback(
    (request: HandoffRequestCreate) => postApi('/cooperative/handoff/request', request, fetchCooperativeState),
    [fetchCooperativeState]
  );

  const fetchMLStatus = useCallback(
    () => fetchApi('/ml/status', setMLStatus), []
  );

  const fetchSwarmStatus = useCallback(
    () => fetchApi('/swarm/status', setSwarmStatus), []
  );

  const configureSwarm = useCallback(
    (config: Partial<SwarmConfig>) => postApi('/swarm/configure', config, fetchSwarmStatus),
    [fetchSwarmStatus]
  );

  const setSwarmFormation = useCallback(
    (formation: FormationType) => postApi('/swarm/formation', { formation }, fetchSwarmStatus),
    [fetchSwarmStatus]
  );

  const fetchHMTStatus = useCallback(
    () => fetchApi('/hmt/status', setHMTStatus), []
  );

  const fetchPendingActions = useCallback(
    () => fetchApi('/hmt/pending', setPendingActions, (d: any) => d.pending || []), []
  );

  const approveAction = useCallback(
    async (actionId: string, reason?: string) => {
      await postApi(`/hmt/approve/${actionId}`, { reason }, async () => {
        await fetchPendingActions();
        await fetchHMTStatus();
      });
    },
    [fetchPendingActions, fetchHMTStatus]
  );

  const rejectAction = useCallback(
    async (actionId: string, reason?: string) => {
      await postApi(`/hmt/reject/${actionId}`, { reason }, async () => {
        await fetchPendingActions();
        await fetchHMTStatus();
      });
    },
    [fetchPendingActions, fetchHMTStatus]
  );

  const setAuthorityLevel = useCallback(
    (level: AuthorityLevel) => postApi('/hmt/authority', { authority_level: level }, fetchHMTStatus),
    [fetchHMTStatus]
  );

  const configureHMT = useCallback(
    (config: Partial<HMTConfig>) => postApi('/hmt/configure', config, fetchHMTStatus),
    [fetchHMTStatus]
  );

  const fetchDatalinkStatus = useCallback(
    () => fetchApi('/datalink/status', setDatalinkStatus), []
  );

  const fetchTerrainStatus = useCallback(
    () => fetchApi('/terrain/status', setTerrainStatus), []
  );

  const fetchPhase7Status = useCallback(
    () => fetchApi('/phase7/status', setPhase7Status), []
  );

  return {
    connected,
    state,
    scenarios,
    guidanceLaws,
    evasionTypes,
    startRun,
    stopRun,
    runMonteCarlo,
    runParameterSweep,
    runEnvelopeAnalysis,
    monteCarloLoading,
    envelopeLoading,
    // Intercept Geometry
    interceptGeometry,
    fetchInterceptGeometry,
    // Threat Assessment
    threatAssessment,
    fetchThreatAssessment,
    // Recording
    isRecording,
    recordings,
    startRecording,
    stopRecording,
    refreshRecordings,
    deleteRecording,
    // Replay
    replayState,
    startReplay,
    pauseReplay,
    resumeReplay,
    seekReplay,
    stopReplay,
    // Sensor Detections
    sensorDetections,
    fetchSensorDetections,
    // WTA
    wtaAlgorithms,
    assignments,
    fetchAssignments,
    costMatrix,
    fetchCostMatrix,
    // Environment
    environmentState,
    fetchEnvironmentState,
    // Kalman & Fusion
    sensorTracks,
    fusedTracks,
    fetchSensorTracks,
    fetchFusedTracks,
    // Cooperative Engagement
    cooperativeState,
    fetchCooperativeState,
    createEngagementZone,
    deleteEngagementZone,
    assignInterceptorToZone,
    requestHandoff,
    // ML
    mlStatus,
    fetchMLStatus,
    // Swarm
    swarmStatus,
    formationTypes,
    fetchSwarmStatus,
    configureSwarm,
    setSwarmFormation,
    // HMT
    hmtStatus,
    authorityLevels,
    pendingActions,
    fetchHMTStatus,
    fetchPendingActions,
    approveAction,
    rejectAction,
    setAuthorityLevel,
    configureHMT,
    // Datalink
    datalinkStatus,
    fetchDatalinkStatus,
    // Terrain
    terrainStatus,
    fetchTerrainStatus,
    // Combined
    phase7Status,
    fetchPhase7Status,
    // Launchers
    launchers,
  };
}
