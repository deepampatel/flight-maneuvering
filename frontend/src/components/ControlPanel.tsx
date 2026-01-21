/**
 * Control Panel - The Mission Control UI
 *
 * This provides:
 * 1. Scenario selection
 * 2. Guidance law selection (NEW in Phase 2)
 * 3. Start/Stop controls
 * 4. Real-time telemetry display
 * 5. Monte Carlo analysis controls (NEW in Phase 2)
 */

import { useState } from 'react';
import type { SimStateEvent, Scenario, GuidanceLaw, MonteCarloResults } from '../types';

interface ControlPanelProps {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  guidanceLaws: GuidanceLaw[];
  onStart: (options: { scenario: string; guidance: string; navConstant: number }) => void;
  onStop: () => void;
  onRunMonteCarlo: (options: {
    scenario: string;
    guidance: string;
    navConstant: number;
    numRuns: number;
    killRadius: number;
    positionNoiseStd: number;
    velocityNoiseStd: number;
  }) => Promise<MonteCarloResults>;
  monteCarloLoading: boolean;
}

export function ControlPanel({
  connected,
  state,
  scenarios,
  guidanceLaws,
  onStart,
  onStop,
  onRunMonteCarlo,
  monteCarloLoading,
}: ControlPanelProps) {
  const [selectedGuidance, setSelectedGuidance] = useState('proportional_nav');
  const [navConstant, setNavConstant] = useState(4.0);
  const [mcResults, setMcResults] = useState<MonteCarloResults | null>(null);
  const [showMonteCarlo, setShowMonteCarlo] = useState(false);

  const isRunning = state?.status === 'running';

  const target = state?.entities.find((e) => e.type === 'target');
  const interceptor = state?.entities.find((e) => e.type === 'interceptor');

  const handleStart = (scenario: string) => {
    onStart({ scenario, guidance: selectedGuidance, navConstant });
  };

  const handleRunMonteCarlo = async (scenario: string) => {
    const results = await onRunMonteCarlo({
      scenario,
      guidance: selectedGuidance,
      navConstant,
      numRuns: 100,
      killRadius: 50,
      positionNoiseStd: 50,
      velocityNoiseStd: 5,
    });
    setMcResults(results);
  };

  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className="status-section">
        <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
          <span className="dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Guidance Selection */}
      <div className="controls-section">
        <h3>Guidance Law</h3>
        <select
          value={selectedGuidance}
          onChange={(e) => setSelectedGuidance(e.target.value)}
          disabled={isRunning}
          className="guidance-select"
        >
          {guidanceLaws.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </select>

        {selectedGuidance !== 'pure_pursuit' && (
          <div className="nav-constant-control">
            <label>Nav Constant (N): {navConstant.toFixed(1)}</label>
            <input
              type="range"
              min="1"
              max="8"
              step="0.5"
              value={navConstant}
              onChange={(e) => setNavConstant(parseFloat(e.target.value))}
              disabled={isRunning}
            />
          </div>
        )}
      </div>

      {/* Scenario Selection & Controls */}
      <div className="controls-section">
        <h3>Scenario</h3>
        <div className="scenario-buttons">
          {Object.entries(scenarios).map(([name, info]) => (
            <button
              key={name}
              onClick={() => handleStart(name)}
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
                ).toFixed(1)}{' '}
                m/s^2
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Monte Carlo Section */}
      <div className="telemetry-section">
        <h3>
          Monte Carlo Analysis
          <button
            className="toggle-btn"
            onClick={() => setShowMonteCarlo(!showMonteCarlo)}
          >
            {showMonteCarlo ? '[-]' : '[+]'}
          </button>
        </h3>

        {showMonteCarlo && (
          <div className="monte-carlo-section">
            <p className="mc-description">
              Run 100 simulations with noise to test robustness
            </p>
            <div className="scenario-buttons">
              {Object.keys(scenarios).map((name) => (
                <button
                  key={`mc-${name}`}
                  onClick={() => handleRunMonteCarlo(name)}
                  disabled={monteCarloLoading}
                >
                  {monteCarloLoading ? 'Running...' : `MC: ${name.replace('_', ' ')}`}
                </button>
              ))}
            </div>

            {mcResults && (
              <div className="mc-results">
                <h4>Results ({mcResults.num_runs} runs)</h4>
                <div className="telemetry-grid">
                  <div className="telemetry-item">
                    <label>Intercept Rate</label>
                    <span className={mcResults.intercept_rate > 0.8 ? 'result-intercept' : 'result-missed'}>
                      {(mcResults.intercept_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="telemetry-item">
                    <label>Mean Miss</label>
                    <span>{mcResults.mean_miss_distance.toFixed(1)}m</span>
                  </div>
                  <div className="telemetry-item">
                    <label>Std Dev</label>
                    <span>{mcResults.std_miss_distance.toFixed(1)}m</span>
                  </div>
                  <div className="telemetry-item">
                    <label>Min/Max</label>
                    <span>
                      {mcResults.min_miss_distance.toFixed(0)}/
                      {mcResults.max_miss_distance.toFixed(0)}m
                    </span>
                  </div>
                </div>

                {/* Simple histogram visualization */}
                <div className="histogram">
                  <h4>Miss Distance Distribution</h4>
                  <div className="histogram-bars">
                    {mcResults.miss_distance_histogram.counts.map((count, i) => {
                      const maxCount = Math.max(...mcResults.miss_distance_histogram.counts);
                      const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                      return (
                        <div
                          key={i}
                          className="histogram-bar"
                          style={{ height: `${height}%` }}
                          title={`${mcResults.miss_distance_histogram.bin_edges[i].toFixed(0)}-${mcResults.miss_distance_histogram.bin_edges[i + 1].toFixed(0)}m: ${count} runs`}
                        />
                      );
                    })}
                  </div>
                  <div className="histogram-labels">
                    <span>0m</span>
                    <span>{mcResults.max_miss_distance.toFixed(0)}m</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

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
          <li>
            <span className="legend-dot target" /> Target
          </li>
          <li>
            <span className="legend-dot interceptor" /> Interceptor
          </li>
          <li>Grid: 1 square = 1km</li>
        </ul>
      </div>
    </div>
  );
}
