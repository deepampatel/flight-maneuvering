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
} from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

interface RunOptions {
  scenario: string;
  guidance: string;
  navConstant: number;
  evasion?: string;
  numInterceptors?: number;
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
}

export function useSimulation(): UseSimulationReturn {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SimStateEvent | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, Scenario>>({});
  const [guidanceLaws, setGuidanceLaws] = useState<GuidanceLaw[]>([]);
  const [evasionTypes, setEvasionTypes] = useState<EvasionType[]>([]);
  const [monteCarloLoading, setMonteCarloLoading] = useState(false);
  const [envelopeLoading, setEnvelopeLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

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

    const response = await fetch(`${API_URL}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario: options.scenario,
        guidance: options.guidance,
        nav_constant: options.navConstant,
        evasion: options.evasion || 'none',
        num_interceptors: options.numInterceptors || 1,
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
  };
}
