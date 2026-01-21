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
} from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

interface RunOptions {
  scenario: string;
  guidance: string;
  navConstant: number;
  evasion?: string;
  numInterceptors?: number;
  numTargets?: number;  // Phase 5: Multi-target support
  wtaAlgorithm?: string;  // Phase 5: WTA algorithm selection
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
  };
}
