/**
 * Control Panel - Mission Control Toolbar
 *
 * Compact horizontal layout with:
 * 1. Main controls in header (scenario, guidance, evasion, interceptors)
 * 2. Floating telemetry HUD overlay
 * 3. Expandable advanced panel for Monte Carlo, Envelope, etc.
 */

import { useState, useEffect } from 'react';
import type {
  SimStateEvent,
  Scenario,
  GuidanceLaw,
  EvasionType,
  MonteCarloResults,
  EnvelopeResults,
  InterceptGeometry,
  ThreatAssessment,
  RecordingMetadata,
  ReplayState,
} from '../types';

interface ControlPanelProps {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  guidanceLaws: GuidanceLaw[];
  evasionTypes: EvasionType[];
  onStart: (options: {
    scenario: string;
    guidance: string;
    navConstant: number;
    evasion: string;
    numInterceptors: number;
  }) => void;
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
  onRunEnvelope: (config: {
    guidance: string;
    nav_constant: number;
    evasion: string;
    range_steps: number;
    bearing_steps: number;
    runs_per_point: number;
  }) => Promise<EnvelopeResults>;
  monteCarloLoading: boolean;
  envelopeLoading: boolean;
  interceptGeometry: InterceptGeometry[] | null;
  threatAssessment: ThreatAssessment[] | null;
  onFetchInterceptGeometry: () => void;
  onFetchThreatAssessment: () => void;
  isRecording: boolean;
  recordings: RecordingMetadata[];
  onStartRecording: () => void;
  onStopRecording: () => void;
  onDeleteRecording: (id: string) => void;
  replayState: ReplayState | null;
  onStartReplay: (id: string) => void;
  onPauseReplay: () => void;
  onResumeReplay: () => void;
  onStopReplay: () => void;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
}

export function ControlPanel({
  connected,
  state,
  scenarios,
  guidanceLaws,
  evasionTypes,
  onStart,
  onStop,
  onRunMonteCarlo,
  onRunEnvelope,
  monteCarloLoading,
  envelopeLoading,
  interceptGeometry,
  threatAssessment,
  onFetchInterceptGeometry,
  onFetchThreatAssessment,
  isRecording,
  recordings,
  onStartRecording,
  onStopRecording,
  onDeleteRecording,
  replayState,
  onStartReplay,
  onPauseReplay,
  onResumeReplay,
  onStopReplay,
  showAdvanced,
  onToggleAdvanced,
}: ControlPanelProps) {
  const [selectedScenario, setSelectedScenario] = useState('head_on');
  const [selectedGuidance, setSelectedGuidance] = useState('proportional_nav');
  const [navConstant, setNavConstant] = useState(4.0);
  const [selectedEvasion, setSelectedEvasion] = useState('none');
  const [numInterceptors, setNumInterceptors] = useState(1);
  const [mcResults, setMcResults] = useState<MonteCarloResults | null>(null);
  const [envelopeResults, setEnvelopeResults] = useState<EnvelopeResults | null>(null);
  const [activePanel, setActivePanel] = useState<string | null>(null);

  const isRunning = state?.status === 'running';
  const target = state?.entities.find((e) => e.type === 'target');
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];

  // Auto-fetch geometry data during simulation
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        onFetchInterceptGeometry();
        onFetchThreatAssessment();
      }, 200);
      return () => clearInterval(interval);
    }
  }, [isRunning, onFetchInterceptGeometry, onFetchThreatAssessment]);

  const handleStart = () => {
    onStart({
      scenario: selectedScenario,
      guidance: selectedGuidance,
      navConstant,
      evasion: selectedEvasion,
      numInterceptors,
    });
  };

  const handleRunMonteCarlo = async () => {
    const results = await onRunMonteCarlo({
      scenario: selectedScenario,
      guidance: selectedGuidance,
      navConstant,
      numRuns: 100,
      killRadius: 50,
      positionNoiseStd: 50,
      velocityNoiseStd: 5,
    });
    setMcResults(results);
  };

  const handleRunEnvelope = async () => {
    const results = await onRunEnvelope({
      guidance: selectedGuidance,
      nav_constant: navConstant,
      evasion: selectedEvasion,
      range_steps: 8,
      bearing_steps: 10,
      runs_per_point: 5,
    });
    setEnvelopeResults(results);
  };

  const togglePanel = (panel: string) => {
    setActivePanel(activePanel === panel ? null : panel);
  };

  return (
    <>
      {/* Main Toolbar Controls */}
      <div className="mission-toolbar">
        <div className="toolbar-group">
          <label>SCENARIO</label>
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
            disabled={isRunning}
          >
            {Object.entries(scenarios).map(([name, info]) => (
              <option key={name} value={name} title={info.description}>
                {name.replace('_', ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        <div className="toolbar-group">
          <label>GUIDANCE</label>
          <select
            value={selectedGuidance}
            onChange={(e) => setSelectedGuidance(e.target.value)}
            disabled={isRunning}
          >
            {guidanceLaws.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}
              </option>
            ))}
          </select>
        </div>

        {selectedGuidance !== 'pure_pursuit' && (
          <div className="toolbar-group nav-group">
            <label>N={navConstant.toFixed(1)}</label>
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

        <div className="toolbar-group">
          <label>EVASION</label>
          <select
            value={selectedEvasion}
            onChange={(e) => setSelectedEvasion(e.target.value)}
            disabled={isRunning}
          >
            {evasionTypes.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </select>
        </div>

        <div className="toolbar-group interceptor-group">
          <label>INT: {numInterceptors}</label>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={numInterceptors}
            onChange={(e) => setNumInterceptors(parseInt(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="toolbar-actions">
          {!isRunning ? (
            <button className="btn-launch" onClick={handleStart} disabled={!connected}>
              LAUNCH
            </button>
          ) : (
            <button className="btn-abort" onClick={onStop}>
              ABORT
            </button>
          )}

          <button
            className={`btn-record ${isRecording ? 'recording' : ''}`}
            onClick={isRecording ? onStopRecording : onStartRecording}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            {isRecording ? 'REC' : 'REC'}
          </button>

          <button
            className={`btn-advanced ${showAdvanced ? 'active' : ''}`}
            onClick={onToggleAdvanced}
          >
            ADV
          </button>
        </div>
      </div>

      {/* Telemetry HUD - Bottom overlay */}
      <div className="telemetry-hud">
        {/* Sim Status */}
        <div className="hud-panel status-panel">
          <div className="hud-title">MISSION</div>
          <div className="hud-content">
            <div className="hud-row">
              <span className="hud-label">T+</span>
              <span className="hud-value">{state ? state.sim_time.toFixed(1) : '0.0'}s</span>
            </div>
            <div className="hud-row">
              <span className="hud-label">STATUS</span>
              <span className={`hud-value status-${state?.status || 'ready'}`}>
                {state?.status?.toUpperCase() || 'READY'}
              </span>
            </div>
            {state?.result && state.result !== 'pending' && (
              <div className="hud-row">
                <span className="hud-label">RESULT</span>
                <span className={`hud-value result-${state.result}`}>
                  {state.result.toUpperCase()}
                </span>
              </div>
            )}
            <div className="hud-row">
              <span className="hud-label">MISS</span>
              <span className="hud-value">{state ? state.miss_distance.toFixed(0) : '---'}m</span>
            </div>
          </div>
        </div>

        {/* Target Telemetry */}
        {target && (
          <div className="hud-panel target-panel">
            <div className="hud-title target">TGT</div>
            <div className="hud-content">
              <div className="hud-row">
                <span className="hud-label">POS</span>
                <span className="hud-value mono">
                  {target.position.x.toFixed(0)}, {target.position.y.toFixed(0)}
                </span>
              </div>
              <div className="hud-row">
                <span className="hud-label">SPD</span>
                <span className="hud-value">{target.speed.toFixed(0)} m/s</span>
              </div>
            </div>
          </div>
        )}

        {/* Interceptor Telemetry */}
        {interceptors.slice(0, 2).map((int, idx) => (
          <div key={int.id} className="hud-panel interceptor-panel">
            <div className="hud-title interceptor">INT{idx + 1}</div>
            <div className="hud-content">
              <div className="hud-row">
                <span className="hud-label">POS</span>
                <span className="hud-value mono">
                  {int.position.x.toFixed(0)}, {int.position.y.toFixed(0)}
                </span>
              </div>
              <div className="hud-row">
                <span className="hud-label">SPD</span>
                <span className="hud-value">{int.speed.toFixed(0)} m/s</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">ACC</span>
                <span className="hud-value">
                  {Math.sqrt(
                    int.acceleration.x ** 2 + int.acceleration.y ** 2 + int.acceleration.z ** 2
                  ).toFixed(0)} m/s²
                </span>
              </div>
            </div>
          </div>
        ))}

        {/* Geometry Panel */}
        {interceptGeometry && interceptGeometry.length > 0 && (
          <div className="hud-panel geometry-panel">
            <div className="hud-title">GEOM</div>
            <div className="hud-content">
              <div className="hud-row">
                <span className="hud-label">RNG</span>
                <span className="hud-value">{(interceptGeometry[0].los_range / 1000).toFixed(2)} km</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">TTI</span>
                <span className="hud-value">
                  {interceptGeometry[0].time_to_intercept >= 0
                    ? `${interceptGeometry[0].time_to_intercept.toFixed(1)}s`
                    : '---'}
                </span>
              </div>
              <div className="hud-row">
                <span className="hud-label">Vc</span>
                <span className="hud-value">{interceptGeometry[0].closing_velocity.toFixed(0)} m/s</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">COL</span>
                <span className={`hud-value ${interceptGeometry[0].collision_course ? 'result-intercept' : 'result-missed'}`}>
                  {interceptGeometry[0].collision_course ? 'YES' : 'NO'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Threat Panel */}
        {threatAssessment && threatAssessment.length > 0 && threatAssessment[0].threats.length > 0 && (
          <div className="hud-panel threat-panel">
            <div className={`hud-title threat-${threatAssessment[0].threats[0].threat_level}`}>
              THREAT
            </div>
            <div className="hud-content">
              <div className="hud-row">
                <span className="hud-label">LVL</span>
                <span className={`hud-value threat-${threatAssessment[0].threats[0].threat_level}`}>
                  {threatAssessment[0].threats[0].threat_level.toUpperCase()}
                </span>
              </div>
              <div className="hud-row">
                <span className="hud-label">SCR</span>
                <span className="hud-value">{threatAssessment[0].threats[0].total_score.toFixed(0)}</span>
              </div>
              <div className="hud-row">
                <span className="hud-label">REC</span>
                <span className="hud-value">{threatAssessment[0].engagement_recommendation.toUpperCase()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Replay Controls */}
        {replayState && (
          <div className="hud-panel replay-panel">
            <div className="hud-title">REPLAY</div>
            <div className="hud-content">
              <div className="replay-progress">
                {replayState.current_tick}/{replayState.total_ticks}
              </div>
              <div className="replay-controls">
                {replayState.is_paused ? (
                  <button onClick={onResumeReplay} className="btn-small">PLAY</button>
                ) : (
                  <button onClick={onPauseReplay} className="btn-small">PAUSE</button>
                )}
                <button onClick={onStopReplay} className="btn-small">STOP</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Advanced Panel - Slide out */}
      {showAdvanced && (
        <div className="advanced-panel">
          <div className="advanced-tabs">
            <button
              className={activePanel === 'montecarlo' ? 'active' : ''}
              onClick={() => togglePanel('montecarlo')}
            >
              Monte Carlo
            </button>
            <button
              className={activePanel === 'envelope' ? 'active' : ''}
              onClick={() => togglePanel('envelope')}
            >
              Envelope
            </button>
            <button
              className={activePanel === 'recordings' ? 'active' : ''}
              onClick={() => togglePanel('recordings')}
            >
              Recordings ({recordings.length})
            </button>
          </div>

          {/* Monte Carlo Content */}
          {activePanel === 'montecarlo' && (
            <div className="advanced-content">
              <p className="panel-desc">Run 100 simulations with noise to test robustness</p>
              <button
                onClick={handleRunMonteCarlo}
                disabled={monteCarloLoading}
                className="btn-action"
              >
                {monteCarloLoading ? 'Running...' : 'Run Monte Carlo'}
              </button>

              {mcResults && (
                <div className="mc-results">
                  <div className="results-grid">
                    <div className="result-item">
                      <span className="result-label">Pk</span>
                      <span className={mcResults.intercept_rate > 0.8 ? 'result-good' : 'result-bad'}>
                        {(mcResults.intercept_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="result-item">
                      <span className="result-label">Mean</span>
                      <span>{mcResults.mean_miss_distance.toFixed(1)}m</span>
                    </div>
                    <div className="result-item">
                      <span className="result-label">StdDev</span>
                      <span>{mcResults.std_miss_distance.toFixed(1)}m</span>
                    </div>
                    <div className="result-item">
                      <span className="result-label">Range</span>
                      <span>{mcResults.min_miss_distance.toFixed(0)}-{mcResults.max_miss_distance.toFixed(0)}m</span>
                    </div>
                  </div>
                  <div className="histogram">
                    {mcResults.miss_distance_histogram.counts.map((count, i) => {
                      const maxCount = Math.max(...mcResults.miss_distance_histogram.counts);
                      const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                      return (
                        <div
                          key={i}
                          className="hist-bar"
                          style={{ height: `${height}%` }}
                          title={`${mcResults.miss_distance_histogram.bin_edges[i].toFixed(0)}-${mcResults.miss_distance_histogram.bin_edges[i + 1].toFixed(0)}m: ${count}`}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Envelope Content */}
          {activePanel === 'envelope' && (
            <div className="advanced-content">
              <p className="panel-desc">Compute intercept probability across range and bearing</p>
              <button
                onClick={handleRunEnvelope}
                disabled={envelopeLoading}
                className="btn-action"
              >
                {envelopeLoading ? 'Computing...' : 'Compute Envelope'}
              </button>

              {envelopeResults && (
                <div className="envelope-results">
                  <div className="heatmap">
                    {envelopeResults.heatmap_2d.data.map((row, ri) => (
                      <div key={ri} className="heatmap-row">
                        {row.map((value, ci) => {
                          const hue = value * 120;
                          return (
                            <div
                              key={ci}
                              className="heatmap-cell"
                              style={{ backgroundColor: `hsl(${hue}, 80%, 40%)` }}
                              title={`R:${envelopeResults.range_values[ri].toFixed(0)}m B:${envelopeResults.bearing_values[ci].toFixed(0)}° Pk:${(value * 100).toFixed(0)}%`}
                            />
                          );
                        })}
                      </div>
                    ))}
                  </div>
                  <div className="heatmap-legend">
                    <span>0%</span>
                    <div className="gradient-bar" />
                    <span>100%</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recordings Content */}
          {activePanel === 'recordings' && (
            <div className="advanced-content">
              {recordings.length > 0 ? (
                <div className="recordings-list">
                  {recordings.slice(0, 8).map((rec) => (
                    <div key={rec.recording_id} className="recording-row">
                      <div className="rec-info">
                        <span className="rec-name">{rec.scenario_name}</span>
                        <span className={`rec-result result-${rec.result}`}>{rec.result}</span>
                        <span className="rec-time">{rec.total_sim_time.toFixed(1)}s</span>
                      </div>
                      <div className="rec-actions">
                        <button onClick={() => onStartReplay(rec.recording_id)}>PLAY</button>
                        <button onClick={() => onDeleteRecording(rec.recording_id)}>DEL</button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="panel-desc">No recordings. Start recording during simulation.</p>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
