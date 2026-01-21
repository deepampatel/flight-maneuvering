/**
 * Control Panel - The Mission Control UI
 *
 * This provides:
 * 1. Scenario selection
 * 2. Start/Stop controls
 * 3. Real-time telemetry display
 * 4. Status indicators
 */

import type { SimStateEvent, Scenario } from '../types';

interface ControlPanelProps {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  onStart: (scenario: string) => void;
  onStop: () => void;
}

export function ControlPanel({
  connected,
  state,
  scenarios,
  onStart,
  onStop,
}: ControlPanelProps) {
  const isRunning = state?.status === 'running';
  const isComplete = state?.status === 'completed';

  const target = state?.entities.find((e) => e.type === 'target');
  const interceptor = state?.entities.find((e) => e.type === 'interceptor');

  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className="status-section">
        <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
          <span className="dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Scenario Selection & Controls */}
      <div className="controls-section">
        <h3>Scenario</h3>
        <div className="scenario-buttons">
          {Object.entries(scenarios).map(([name, info]) => (
            <button
              key={name}
              onClick={() => onStart(name)}
              disabled={!connected || isRunning}
              title={info.description}
            >
              {name.replace('_', ' ').toUpperCase()}
            </button>
          ))}
        </div>

        {isRunning && (
          <button className="stop-button" onClick={onStop}>
            STOP
          </button>
        )}
      </div>

      {/* Simulation Status */}
      {state && (
        <div className="telemetry-section">
          <h3>Status</h3>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <label>Run ID</label>
              <span>{state.run_id}</span>
            </div>
            <div className="telemetry-item">
              <label>Sim Time</label>
              <span>{state.sim_time.toFixed(2)}s</span>
            </div>
            <div className="telemetry-item">
              <label>Status</label>
              <span className={`status-${state.status}`}>
                {state.status.toUpperCase()}
              </span>
            </div>
            {state.result !== 'pending' && (
              <div className="telemetry-item">
                <label>Result</label>
                <span className={`result-${state.result}`}>
                  {state.result.toUpperCase()}
                </span>
              </div>
            )}
            <div className="telemetry-item">
              <label>Miss Distance</label>
              <span>{state.miss_distance.toFixed(1)}m</span>
            </div>
          </div>
        </div>
      )}

      {/* Entity Telemetry */}
      {target && (
        <div className="telemetry-section">
          <h3>Target (T1)</h3>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <label>Position</label>
              <span>
                ({target.position.x.toFixed(0)}, {target.position.y.toFixed(0)},{' '}
                {target.position.z.toFixed(0)})
              </span>
            </div>
            <div className="telemetry-item">
              <label>Speed</label>
              <span>{target.speed.toFixed(1)} m/s</span>
            </div>
          </div>
        </div>
      )}

      {interceptor && (
        <div className="telemetry-section">
          <h3>Interceptor (I1)</h3>
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <label>Position</label>
              <span>
                ({interceptor.position.x.toFixed(0)}, {interceptor.position.y.toFixed(0)},{' '}
                {interceptor.position.z.toFixed(0)})
              </span>
            </div>
            <div className="telemetry-item">
              <label>Speed</label>
              <span>{interceptor.speed.toFixed(1)} m/s</span>
            </div>
            <div className="telemetry-item">
              <label>Acceleration</label>
              <span>
                {Math.sqrt(
                  interceptor.acceleration.x ** 2 +
                  interceptor.acceleration.y ** 2 +
                  interceptor.acceleration.z ** 2
                ).toFixed(1)} m/s²
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="instructions">
        <h4>Controls</h4>
        <ul>
          <li>Left-click + drag: Rotate view</li>
          <li>Right-click + drag: Pan</li>
          <li>Scroll: Zoom</li>
        </ul>
        <h4>Legend</h4>
        <ul>
          <li><span className="legend-dot target" /> Target</li>
          <li><span className="legend-dot interceptor" /> Interceptor</li>
          <li>Grid: 1 square = 1km</li>
        </ul>
      </div>
    </div>
  );
}
