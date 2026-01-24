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
  // Phase 5
  SensorDetection,
  AssignmentResult,
  WTAAlgorithm,
  CostMatrix,
  // Phase 6
  EnvironmentState,
  SensorTrack,
  SensorTracksResponse,
  FusedTrack,
  FusedTracksResponse,
  CooperativeState,
  EngagementZoneCreateRequest,
  HandoffRequestCreate,
  // Phase 6.4: ML
  MLStatus,
  // Phase 7
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
} from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

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

interface RunOptions {
  scenario: string;
  guidance: string;
  navConstant: number;
  evasion?: string;
  numInterceptors?: number;
  numTargets?: number;  // Phase 5: Multi-target support
  wtaAlgorithm?: string;  // Phase 5: WTA algorithm selection
  // Phase 6: Environment
  windSpeed?: number;
  windDirection?: number;
  windGusts?: number;
  enableDrag?: boolean;
  // Phase 6: Cooperative
  enableCooperative?: boolean;
  // Mission Planner: Custom entities
  customEntities?: PlannedEntity[];
  customZones?: PlannedZone[];
  // Phase 7: Swarm
  enableSwarm?: boolean;
  swarmFormation?: FormationType;
  swarmSpacing?: number;
  // Phase 7: HMT
  enableHmt?: boolean;
  hmtAuthorityLevel?: AuthorityLevel;
  // Phase 7: Datalink
  enableDatalink?: boolean;
  // Phase 7: Terrain
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
  // Phase 4: Intercept Geometry
  interceptGeometry: InterceptGeometry[] | null;
  fetchInterceptGeometry: () => Promise<void>;
  // Phase 4: Threat Assessment
  threatAssessment: ThreatAssessment[] | null;
  fetchThreatAssessment: () => Promise<void>;
  // Phase 4: Recording
  isRecording: boolean;
  recordings: RecordingMetadata[];
  startRecording: () => Promise<string>;
  stopRecording: () => Promise<void>;
  refreshRecordings: () => Promise<void>;
  deleteRecording: (recordingId: string) => Promise<void>;
  // Phase 4: Replay
  replayState: ReplayState | null;
  startReplay: (recordingId: string, config?: Partial<ReplayConfig>) => Promise<void>;
  pauseReplay: () => Promise<void>;
  resumeReplay: () => Promise<void>;
  seekReplay: (tick: number) => Promise<void>;
  stopReplay: () => Promise<void>;
  // Phase 5: Sensor Detections
  sensorDetections: Record<string, SensorDetection[]> | null;
  fetchSensorDetections: () => Promise<void>;
  // Phase 5: WTA
  wtaAlgorithms: WTAAlgorithm[];
  assignments: AssignmentResult | null;
  fetchAssignments: (algorithm?: string) => Promise<void>;
  costMatrix: CostMatrix | null;
  fetchCostMatrix: () => Promise<void>;
  // Phase 6: Environment
  environmentState: EnvironmentState | null;
  fetchEnvironmentState: () => Promise<void>;
  // Phase 6: Kalman & Fusion
  sensorTracks: SensorTrack[] | null;
  fusedTracks: FusedTrack[] | null;
  fetchSensorTracks: () => Promise<void>;
  fetchFusedTracks: () => Promise<void>;
  // Phase 6: Cooperative Engagement
  cooperativeState: CooperativeState | null;
  fetchCooperativeState: () => Promise<void>;
  createEngagementZone: (zone: EngagementZoneCreateRequest) => Promise<void>;
  deleteEngagementZone: (zoneId: string) => Promise<void>;
  assignInterceptorToZone: (interceptorId: string, zoneId: string) => Promise<void>;
  requestHandoff: (request: HandoffRequestCreate) => Promise<void>;
  // Phase 6.4: ML
  mlStatus: MLStatus | null;
  fetchMLStatus: () => Promise<void>;
  // Phase 7: Swarm
  swarmStatus: SwarmStatus | null;
  formationTypes: FormationInfo[];
  fetchSwarmStatus: () => Promise<void>;
  configureSwarm: (config: Partial<SwarmConfig>) => Promise<void>;
  setSwarmFormation: (formation: FormationType) => Promise<void>;
  // Phase 7: HMT
  hmtStatus: HMTStatus | null;
  authorityLevels: AuthorityLevelInfo[];
  pendingActions: PendingAction[];
  fetchHMTStatus: () => Promise<void>;
  fetchPendingActions: () => Promise<void>;
  approveAction: (actionId: string, reason?: string) => Promise<void>;
  rejectAction: (actionId: string, reason?: string) => Promise<void>;
  setAuthorityLevel: (level: AuthorityLevel) => Promise<void>;
  configureHMT: (config: Partial<HMTConfig>) => Promise<void>;
  // Phase 7: Datalink
  datalinkStatus: DatalinkStatus | null;
  fetchDatalinkStatus: () => Promise<void>;
  // Phase 7: Terrain
  terrainStatus: TerrainStatus | null;
  fetchTerrainStatus: () => Promise<void>;
  // Phase 7: Combined status
  phase7Status: Phase7Status | null;
  fetchPhase7Status: () => Promise<void>;
}

export function useSimulation(): UseSimulationReturn {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SimStateEvent | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, Scenario>>({});
  const [guidanceLaws, setGuidanceLaws] = useState<GuidanceLaw[]>([]);
  const [evasionTypes, setEvasionTypes] = useState<EvasionType[]>([]);
  const [monteCarloLoading, setMonteCarloLoading] = useState(false);
  const [envelopeLoading, setEnvelopeLoading] = useState(false);
  // Phase 4: Intercept Geometry & Threat Assessment
  const [interceptGeometry, setInterceptGeometry] = useState<InterceptGeometry[] | null>(null);
  const [threatAssessment, setThreatAssessment] = useState<ThreatAssessment[] | null>(null);
  // Phase 4: Recording & Replay
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState<RecordingMetadata[]>([]);
  const [replayState, setReplayState] = useState<ReplayState | null>(null);
  // Phase 5: Sensors & WTA
  const [sensorDetections, setSensorDetections] = useState<Record<string, SensorDetection[]> | null>(null);
  const [wtaAlgorithms, setWtaAlgorithms] = useState<WTAAlgorithm[]>([]);
  const [assignments, setAssignments] = useState<AssignmentResult | null>(null);
  const [costMatrix, setCostMatrix] = useState<CostMatrix | null>(null);
  // Phase 6: Environment
  const [environmentState, setEnvironmentState] = useState<EnvironmentState | null>(null);
  // Phase 6: Kalman & Fusion
  const [sensorTracks, setSensorTracks] = useState<SensorTrack[] | null>(null);
  const [fusedTracks, setFusedTracks] = useState<FusedTrack[] | null>(null);
  // Phase 6: Cooperative Engagement
  const [cooperativeState, setCooperativeState] = useState<CooperativeState | null>(null);
  // Phase 6.4: ML
  const [mlStatus, setMLStatus] = useState<MLStatus | null>(null);
  // Phase 7: Swarm
  const [swarmStatus, setSwarmStatus] = useState<SwarmStatus | null>(null);
  const [formationTypes, setFormationTypes] = useState<FormationInfo[]>([]);
  // Phase 7: HMT
  const [hmtStatus, setHMTStatus] = useState<HMTStatus | null>(null);
  const [authorityLevels, setAuthorityLevels] = useState<AuthorityLevelInfo[]>([]);
  const [pendingActions, setPendingActions] = useState<PendingAction[]>([]);
  // Phase 7: Datalink
  const [datalinkStatus, setDatalinkStatus] = useState<DatalinkStatus | null>(null);
  // Phase 7: Terrain
  const [terrainStatus, setTerrainStatus] = useState<TerrainStatus | null>(null);
  // Phase 7: Combined
  const [phase7Status, setPhase7Status] = useState<Phase7Status | null>(null);

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

    // Phase 5: Fetch WTA algorithms
    fetch(`${API_URL}/wta/algorithms`)
      .then((res) => res.json())
      .then((data) => setWtaAlgorithms(data.algorithms))
      .catch(console.error);

    // Phase 7: Fetch formation types
    fetch(`${API_URL}/swarm/formations`)
      .then((res) => res.json())
      .then((data) => setFormationTypes(data.formations || []))
      .catch(console.error);

    // Phase 7: Fetch authority levels
    fetch(`${API_URL}/hmt/authority-levels`)
      .then((res) => res.json())
      .then((data) => setAuthorityLevels(data.authority_levels || []))
      .catch(console.error);

    // Phase 7: Fetch combined status
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
          // Phase 5: Extract assignments from state event if present
          if (data.assignments) {
            setAssignments(data.assignments);
          }
          // Phase 7: Extract HMT pending actions from state event
          if (data.hmt?.pending_actions) {
            setPendingActions(data.hmt.pending_actions);
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
    // Phase 5: Clear sensor/WTA state
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
        num_targets: options.numTargets,  // Phase 5: Multi-target
        wta_algorithm: options.wtaAlgorithm || 'hungarian',  // Phase 5: WTA algorithm
        real_time: true,
        // Phase 6: Environment
        wind_speed: options.windSpeed || 0,
        wind_direction: options.windDirection || 0,
        wind_gusts: options.windGusts || 0,
        enable_drag: options.enableDrag || false,
        // Phase 6: Cooperative
        enable_cooperative: options.enableCooperative || false,
        // Mission Planner: Custom entities
        custom_entities: options.customEntities,
        custom_zones: options.customZones,
        // Phase 7: Swarm
        enable_swarm: options.enableSwarm || false,
        swarm_formation: options.swarmFormation,
        swarm_spacing: options.swarmSpacing,
        // Phase 7: HMT
        enable_hmt: options.enableHmt || false,
        hmt_authority_level: options.hmtAuthorityLevel,
        // Phase 7: Datalink
        enable_datalink: options.enableDatalink || false,
        // Phase 7: Terrain
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

  // Phase 4: Fetch intercept geometry
  const fetchInterceptGeometry = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/intercept-geometry`);
      if (response.ok) {
        const data = await response.json();
        setInterceptGeometry(data.geometries);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 4: Fetch threat assessment
  const fetchThreatAssessment = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/threat-assessment`);
      if (response.ok) {
        const data = await response.json();
        setThreatAssessment(data.assessments);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 4: Start recording
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

  // Phase 4: Stop recording
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

  // Phase 4: Refresh recordings list
  const refreshRecordings = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/recordings`);
      const data = await response.json();
      setRecordings(data.recordings);
    } catch (e) {
      console.error('Failed to fetch recordings:', e);
    }
  }, []);

  // Phase 4: Delete recording
  const deleteRecording = useCallback(async (recordingId: string) => {
    await fetch(`${API_URL}/recordings/${recordingId}`, {
      method: 'DELETE',
    });
    // Refresh list
    const response = await fetch(`${API_URL}/recordings`);
    const data = await response.json();
    setRecordings(data.recordings);
  }, []);

  // Phase 4: Start replay
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

  // Phase 4: Pause replay
  const pauseReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/pause`, { method: 'POST' });
    // Update replay state
    const response = await fetch(`${API_URL}/replay/state`);
    const data = await response.json();
    if (data.recording_id) {
      setReplayState(data);
    }
  }, []);

  // Phase 4: Resume replay
  const resumeReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/resume`, { method: 'POST' });
    const response = await fetch(`${API_URL}/replay/state`);
    const data = await response.json();
    if (data.recording_id) {
      setReplayState(data);
    }
  }, []);

  // Phase 4: Seek replay
  const seekReplay = useCallback(async (tick: number) => {
    await fetch(`${API_URL}/replay/seek?tick=${tick}`, { method: 'POST' });
  }, []);

  // Phase 4: Stop replay
  const stopReplay = useCallback(async () => {
    await fetch(`${API_URL}/replay/stop`, { method: 'POST' });
    setReplayState(null);
  }, []);

  // Fetch recordings on mount
  useEffect(() => {
    refreshRecordings();
  }, [refreshRecordings]);

  // Phase 5: Fetch sensor detections
  const fetchSensorDetections = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/sensor/detections`);
      if (response.ok) {
        const data = await response.json();
        setSensorDetections(data.detections);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 5: Fetch WTA assignments
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

  // Phase 5: Fetch cost matrix
  const fetchCostMatrix = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/wta/cost-matrix`);
      if (response.ok) {
        const data = await response.json();
        setCostMatrix(data);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 6: Fetch environment state
  const fetchEnvironmentState = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/environment/config`);
      if (response.ok) {
        const data = await response.json();
        setEnvironmentState(data);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 6: Fetch sensor tracks with Kalman state
  const fetchSensorTracks = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/sensor/tracks`);
      if (response.ok) {
        const data: SensorTracksResponse = await response.json();
        // Flatten tracks from all sensors into a single list
        const allTracks: SensorTrack[] = [];
        for (const tracks of Object.values(data.tracks_by_sensor)) {
          allTracks.push(...tracks);
        }
        setSensorTracks(allTracks);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 6: Fetch fused tracks
  const fetchFusedTracks = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/sensor/fused-tracks`);
      if (response.ok) {
        const data: FusedTracksResponse = await response.json();
        setFusedTracks(data.fused_tracks);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 6: Fetch cooperative state
  const fetchCooperativeState = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/cooperative/state`);
      if (response.ok) {
        const data: CooperativeState = await response.json();
        setCooperativeState(data);
      }
    } catch (e) {
      // Silently fail if no active simulation
    }
  }, []);

  // Phase 6: Create engagement zone
  const createEngagementZone = useCallback(async (zone: EngagementZoneCreateRequest) => {
    try {
      const response = await fetch(`${API_URL}/cooperative/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zone),
      });
      if (response.ok) {
        // Refresh cooperative state
        await fetchCooperativeState();
      }
    } catch (e) {
      console.error('Failed to create engagement zone:', e);
    }
  }, [fetchCooperativeState]);

  // Phase 6: Delete engagement zone
  const deleteEngagementZone = useCallback(async (zoneId: string) => {
    try {
      const response = await fetch(`${API_URL}/cooperative/zones/${zoneId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        // Refresh cooperative state
        await fetchCooperativeState();
      }
    } catch (e) {
      console.error('Failed to delete engagement zone:', e);
    }
  }, [fetchCooperativeState]);

  // Phase 6: Assign interceptor to zone
  const assignInterceptorToZone = useCallback(async (interceptorId: string, zoneId: string) => {
    try {
      const response = await fetch(`${API_URL}/cooperative/zones/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          interceptor_id: interceptorId,
          zone_id: zoneId,
        }),
      });
      if (response.ok) {
        // Refresh cooperative state
        await fetchCooperativeState();
      }
    } catch (e) {
      console.error('Failed to assign interceptor to zone:', e);
    }
  }, [fetchCooperativeState]);

  // Phase 6: Request handoff
  const requestHandoff = useCallback(async (request: HandoffRequestCreate) => {
    try {
      const response = await fetch(`${API_URL}/cooperative/handoff/request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      if (response.ok) {
        // Refresh cooperative state
        await fetchCooperativeState();
      }
    } catch (e) {
      console.error('Failed to request handoff:', e);
    }
  }, [fetchCooperativeState]);

  // Phase 6.4: Fetch ML status
  const fetchMLStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/ml/status`);
      if (response.ok) {
        const data: MLStatus = await response.json();
        setMLStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch ML status:', e);
    }
  }, []);

  // Phase 7: Fetch swarm status
  const fetchSwarmStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/swarm/status`);
      if (response.ok) {
        const data: SwarmStatus = await response.json();
        setSwarmStatus(data);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

  // Phase 7: Configure swarm
  const configureSwarm = useCallback(async (config: Partial<SwarmConfig>) => {
    try {
      const response = await fetch(`${API_URL}/swarm/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (response.ok) {
        await fetchSwarmStatus();
      }
    } catch (e) {
      console.error('Failed to configure swarm:', e);
    }
  }, [fetchSwarmStatus]);

  // Phase 7: Set swarm formation
  const setSwarmFormation = useCallback(async (formation: FormationType) => {
    try {
      const response = await fetch(`${API_URL}/swarm/formation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formation }),
      });
      if (response.ok) {
        await fetchSwarmStatus();
      }
    } catch (e) {
      console.error('Failed to set formation:', e);
    }
  }, [fetchSwarmStatus]);

  // Phase 7: Fetch HMT status
  const fetchHMTStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/hmt/status`);
      if (response.ok) {
        const data: HMTStatus = await response.json();
        setHMTStatus(data);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

  // Phase 7: Fetch pending actions
  const fetchPendingActions = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/hmt/pending`);
      if (response.ok) {
        const data = await response.json();
        setPendingActions(data.pending || []);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

  // Phase 7: Approve action
  const approveAction = useCallback(async (actionId: string, reason?: string) => {
    try {
      const response = await fetch(`${API_URL}/hmt/approve/${actionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      });
      if (response.ok) {
        await fetchPendingActions();
        await fetchHMTStatus();
      }
    } catch (e) {
      console.error('Failed to approve action:', e);
    }
  }, [fetchPendingActions, fetchHMTStatus]);

  // Phase 7: Reject action
  const rejectAction = useCallback(async (actionId: string, reason?: string) => {
    try {
      const response = await fetch(`${API_URL}/hmt/reject/${actionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      });
      if (response.ok) {
        await fetchPendingActions();
        await fetchHMTStatus();
      }
    } catch (e) {
      console.error('Failed to reject action:', e);
    }
  }, [fetchPendingActions, fetchHMTStatus]);

  // Phase 7: Set authority level
  const setAuthorityLevel = useCallback(async (level: AuthorityLevel) => {
    try {
      const response = await fetch(`${API_URL}/hmt/authority`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ authority_level: level }),
      });
      if (response.ok) {
        await fetchHMTStatus();
      }
    } catch (e) {
      console.error('Failed to set authority level:', e);
    }
  }, [fetchHMTStatus]);

  // Phase 7: Configure HMT
  const configureHMT = useCallback(async (config: Partial<HMTConfig>) => {
    try {
      const response = await fetch(`${API_URL}/hmt/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (response.ok) {
        await fetchHMTStatus();
      }
    } catch (e) {
      console.error('Failed to configure HMT:', e);
    }
  }, [fetchHMTStatus]);

  // Phase 7: Fetch datalink status
  const fetchDatalinkStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/datalink/status`);
      if (response.ok) {
        const data: DatalinkStatus = await response.json();
        setDatalinkStatus(data);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

  // Phase 7: Fetch terrain status
  const fetchTerrainStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/terrain/status`);
      if (response.ok) {
        const data: TerrainStatus = await response.json();
        setTerrainStatus(data);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

  // Phase 7: Fetch combined Phase 7 status
  const fetchPhase7Status = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/phase7/status`);
      if (response.ok) {
        const data: Phase7Status = await response.json();
        setPhase7Status(data);
      }
    } catch (e) {
      // Silently fail if not available
    }
  }, []);

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
    // Phase 4: Intercept Geometry
    interceptGeometry,
    fetchInterceptGeometry,
    // Phase 4: Threat Assessment
    threatAssessment,
    fetchThreatAssessment,
    // Phase 4: Recording
    isRecording,
    recordings,
    startRecording,
    stopRecording,
    refreshRecordings,
    deleteRecording,
    // Phase 4: Replay
    replayState,
    startReplay,
    pauseReplay,
    resumeReplay,
    seekReplay,
    stopReplay,
    // Phase 5: Sensor Detections
    sensorDetections,
    fetchSensorDetections,
    // Phase 5: WTA
    wtaAlgorithms,
    assignments,
    fetchAssignments,
    costMatrix,
    fetchCostMatrix,
    // Phase 6: Environment
    environmentState,
    fetchEnvironmentState,
    // Phase 6: Kalman & Fusion
    sensorTracks,
    fusedTracks,
    fetchSensorTracks,
    fetchFusedTracks,
    // Phase 6: Cooperative Engagement
    cooperativeState,
    fetchCooperativeState,
    createEngagementZone,
    deleteEngagementZone,
    assignInterceptorToZone,
    requestHandoff,
    // Phase 6.4: ML
    mlStatus,
    fetchMLStatus,
    // Phase 7: Swarm
    swarmStatus,
    formationTypes,
    fetchSwarmStatus,
    configureSwarm,
    setSwarmFormation,
    // Phase 7: HMT
    hmtStatus,
    authorityLevels,
    pendingActions,
    fetchHMTStatus,
    fetchPendingActions,
    approveAction,
    rejectAction,
    setAuthorityLevel,
    configureHMT,
    // Phase 7: Datalink
    datalinkStatus,
    fetchDatalinkStatus,
    // Phase 7: Terrain
    terrainStatus,
    fetchTerrainStatus,
    // Phase 7: Combined
    phase7Status,
    fetchPhase7Status,
  };
}
