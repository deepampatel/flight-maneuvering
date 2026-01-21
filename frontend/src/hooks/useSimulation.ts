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
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { SimStateEvent, SimEvent, Scenario } from '../types';

const WS_URL = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';

interface UseSimulationReturn {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  startRun: (scenario: string) => Promise<void>;
  stopRun: () => Promise<void>;
}

export function useSimulation(): UseSimulationReturn {
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<SimStateEvent | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, Scenario>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

  // Fetch available scenarios on mount
  useEffect(() => {
    fetch(`${API_URL}/scenarios`)
      .then((res) => res.json())
      .then(setScenarios)
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
  const startRun = useCallback(async (scenario: string) => {
    setState(null); // Clear previous state

    const response = await fetch(`${API_URL}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario, real_time: true }),
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

  return {
    connected,
    state,
    scenarios,
    startRun,
    stopRun,
  };
}
