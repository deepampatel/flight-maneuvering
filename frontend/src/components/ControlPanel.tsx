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
  // Phase 5
  WTAAlgorithm,
  AssignmentResult,
  // Phase 6
  EnvironmentState,
  CooperativeState,
  EngagementZoneCreateRequest,
  HandoffRequestCreate,
  // Phase 6.4: ML
  MLStatus,
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
    numTargets?: number;  // Phase 5
    wtaAlgorithm?: string;  // Phase 5
    // Phase 6: Environment
    windSpeed?: number;
    windDirection?: number;
    windGusts?: number;
    enableDrag?: boolean;
    // Phase 6: Cooperative
    enableCooperative?: boolean;
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
  // Phase 5: WTA
  wtaAlgorithms: WTAAlgorithm[];
  assignments: AssignmentResult | null;
  onFetchAssignments: (algorithm?: string) => void;
  // Phase 6: Environment
  environmentState: EnvironmentState | null;
  // Phase 6: Sensor tracks
  onFetchSensorTracks?: () => void;
  // Phase 6: Cooperative Engagement
  cooperativeState?: CooperativeState | null;
  onFetchCooperativeState?: () => void;
  onCreateEngagementZone?: (zone: EngagementZoneCreateRequest) => void;
  onDeleteEngagementZone?: (zoneId: string) => void;
  onAssignInterceptorToZone?: (interceptorId: string, zoneId: string) => void;
  onRequestHandoff?: (request: HandoffRequestCreate) => void;
  // Mission Planner (props passed through but not used in ControlPanel)
  plannerMode?: string;
  onSetPlannerMode?: (mode: string) => void;
  plannedEntities?: { id: string; type: string; position: { x: number; y: number; z: number }; velocity: { x: number; y: number; z: number } }[];
  plannedZones?: { id: string; name: string; center: { x: number; y: number; z: number }; dimensions: { x: number; y: number; z: number }; color: string }[];
  onClearPlanner?: () => void;
  onRemovePlannedEntity?: (id: string) => void;
  // Phase 6.4: ML
  mlStatus?: MLStatus | null;
  onFetchMLStatus?: () => void;
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
  // Phase 5
  wtaAlgorithms,
  assignments,
  onFetchAssignments,
  // Phase 6
  environmentState,
  onFetchSensorTracks,
  // Phase 6: Cooperative
  cooperativeState,
  onFetchCooperativeState,
  onCreateEngagementZone,
  onDeleteEngagementZone: _onDeleteEngagementZone,  // Reserved for future UI
  onAssignInterceptorToZone: _onAssignInterceptorToZone,  // Reserved for future UI
  onRequestHandoff: _onRequestHandoff,  // Reserved for future UI
  // Mission Planner (not used in ControlPanel, passed from App)
  plannerMode: _plannerMode,
  onSetPlannerMode: _onSetPlannerMode,
  plannedEntities: _plannedEntities,
  plannedZones: _plannedZones,
  onClearPlanner: _onClearPlanner,
  onRemovePlannedEntity: _onRemovePlannedEntity,
  // Phase 6.4: ML
  mlStatus,
  onFetchMLStatus,
}: ControlPanelProps) {
  // Suppress unused warnings for reserved handlers
  void _onDeleteEngagementZone;
  void _onAssignInterceptorToZone;
  void _onRequestHandoff;
  void _plannerMode;
  void _onSetPlannerMode;
  void _plannedEntities;
  void _plannedZones;
  void _onClearPlanner;
  void _onRemovePlannedEntity;
  const [selectedScenario, setSelectedScenario] = useState('head_on');
  const [selectedGuidance, setSelectedGuidance] = useState('proportional_nav');
  const [navConstant, setNavConstant] = useState(4.0);
  const [selectedEvasion, setSelectedEvasion] = useState('none');
  const [numInterceptors, setNumInterceptors] = useState(1);
  const [numTargets, setNumTargets] = useState(1);  // Phase 5
  const [selectedWTA, setSelectedWTA] = useState('hungarian');  // Phase 5: Default to optimal
  // Phase 6: Environment state
  const [windSpeed, setWindSpeed] = useState(0);  // m/s
  const [windDirection, setWindDirection] = useState(0);  // degrees
  const [windGusts, setWindGusts] = useState(0);  // m/s
  const [enableDrag, setEnableDrag] = useState(false);
  // Phase 6: Cooperative state
  const [enableCooperative, setEnableCooperative] = useState(false);
  const [mcResults, setMcResults] = useState<MonteCarloResults | null>(null);
  const [envelopeResults, setEnvelopeResults] = useState<EnvelopeResults | null>(null);
  const [activePanel, setActivePanel] = useState<string | null>(null);

  const isRunning = state?.status === 'running';
  const targets = state?.entities.filter((e) => e.type === 'target') || [];  // Phase 5
  const target = targets[0];  // Backward compat
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];

  // Auto-fetch geometry data during simulation
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        onFetchInterceptGeometry();
        onFetchThreatAssessment();
        // Phase 5: Also fetch WTA assignments for multi-target scenarios
        if (numTargets > 1) {
          onFetchAssignments(selectedWTA);
        }
        // Phase 6: Fetch sensor tracks for uncertainty visualization
        if (onFetchSensorTracks) {
          onFetchSensorTracks();
        }
        // Phase 6: Fetch cooperative state
        if (onFetchCooperativeState && enableCooperative) {
          onFetchCooperativeState();
        }
      }, 200);
      return () => clearInterval(interval);
    }
  }, [isRunning, onFetchInterceptGeometry, onFetchThreatAssessment, onFetchAssignments, numTargets, selectedWTA, onFetchSensorTracks, onFetchCooperativeState, enableCooperative]);

  // Phase 5: Update numTargets when scenario changes
  useEffect(() => {
    const scenario = scenarios[selectedScenario];
    if (scenario?.num_targets) {
      setNumTargets(scenario.num_targets);
    }
  }, [selectedScenario, scenarios]);

  const handleStart = () => {
    onStart({
      scenario: selectedScenario,
      guidance: selectedGuidance,
      navConstant,
      evasion: selectedEvasion,
      numInterceptors,
      numTargets,  // Phase 5
      wtaAlgorithm: selectedWTA,  // Phase 5
      // Phase 6: Environment
      windSpeed,
      windDirection,
      windGusts,
      enableDrag,
      // Phase 6: Cooperative
      enableCooperative,
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

        {/* Phase 5: Targets slider */}
        <div className="toolbar-group target-group">
          <label>TGT: {numTargets}</label>
          <input
            type="range"
            min="1"
            max="4"
            step="1"
            value={numTargets}
            onChange={(e) => setNumTargets(parseInt(e.target.value))}
            disabled={isRunning}
          />
        </div>

        {/* Phase 5: WTA algorithm selector (show when multiple targets) */}
        {numTargets > 1 && (
          <div className="toolbar-group">
            <label>WTA</label>
            <select
              value={selectedWTA}
              onChange={(e) => setSelectedWTA(e.target.value)}
              disabled={isRunning}
            >
              {wtaAlgorithms.map((alg) => (
                <option key={alg.id} value={alg.id} title={alg.description}>
                  {alg.name}
                </option>
              ))}
            </select>
          </div>
        )}

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

        {/* Interceptor Telemetry with assigned target geometry */}
        {interceptors.slice(0, 4).map((int) => {
          // Find the assignment for this interceptor
          const assignment = assignments?.assignments.find(a => a.interceptor_id === int.id);
          const assignedTargetId = assignment?.target_id;

          // Find geometry for this interceptor's assigned target only
          const geom = interceptGeometry?.find(
            g => g.interceptor_id === int.id &&
                 (assignedTargetId ? g.target_id === assignedTargetId : true)
          );

          // Check if this interceptor has hit its target
          const hasHit = state?.intercepted_pairs?.some(pair => pair[0] === int.id);

          return (
            <div key={int.id} className={`hud-panel interceptor-panel ${hasHit ? 'interceptor-hit' : ''}`}>
              <div className="hud-title interceptor">
                {int.id}
                {assignedTargetId && <span className="assigned-target">→{assignedTargetId}</span>}
              </div>
              <div className="hud-content">
                {hasHit ? (
                  <div className="hud-row">
                    <span className="hud-label">STATUS</span>
                    <span className="hud-value result-intercept">HIT</span>
                  </div>
                ) : (
                  <>
                    <div className="hud-row">
                      <span className="hud-label">SPD</span>
                      <span className="hud-value">{int.speed.toFixed(0)} m/s</span>
                    </div>
                    {geom && (
                      <>
                        <div className="hud-row">
                          <span className="hud-label">RNG</span>
                          <span className="hud-value">{(geom.los_range / 1000).toFixed(2)} km</span>
                        </div>
                        <div className="hud-row">
                          <span className="hud-label">TTI</span>
                          <span className="hud-value">
                            {geom.time_to_intercept >= 0 ? `${geom.time_to_intercept.toFixed(1)}s` : '---'}
                          </span>
                        </div>
                        <div className="hud-row">
                          <span className="hud-label">Vc</span>
                          <span className="hud-value">{geom.closing_velocity.toFixed(0)} m/s</span>
                        </div>
                        <div className="hud-row">
                          <span className="hud-label">COL</span>
                          <span className={`hud-value ${geom.collision_course ? 'result-intercept' : 'result-missed'}`}>
                            {geom.collision_course ? 'YES' : 'NO'}
                          </span>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          );
        })}

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

        {/* Phase 5: WTA Assignments Panel */}
        {assignments && assignments.assignments.length > 0 && targets.length > 1 && (
          <div className="hud-panel wta-panel">
            <div className="hud-title">WTA</div>
            <div className="hud-content">
              <div className="hud-row">
                <span className="hud-label">ALGO</span>
                <span className="hud-value">{assignments.algorithm.split('_').join(' ').toUpperCase()}</span>
              </div>
              {assignments.assignments.slice(0, 3).map((a) => (
                <div key={a.interceptor_id} className="hud-row">
                  <span className="hud-label">{a.interceptor_id}</span>
                  <span className="hud-value">{a.target_id}</span>
                </div>
              ))}
              {assignments.unassigned_targets.length > 0 && (
                <div className="hud-row">
                  <span className="hud-label">UNASGN</span>
                  <span className="hud-value result-missed">{assignments.unassigned_targets.join(', ')}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Phase 5: Multi-target summary */}
        {targets.length > 1 && state?.intercepted_pairs && state.intercepted_pairs.length > 0 && (
          <div className="hud-panel intercepts-panel">
            <div className="hud-title result-intercept">KILLS</div>
            <div className="hud-content">
              {state.intercepted_pairs.map(([intId, tgtId]) => (
                <div key={`${intId}-${tgtId}`} className="hud-row">
                  <span className="hud-label">{intId}</span>
                  <span className="hud-value result-intercept">{tgtId}</span>
                </div>
              ))}
              <div className="hud-row">
                <span className="hud-label">TOTAL</span>
                <span className="hud-value">{state.intercepted_pairs.length}/{targets.length}</span>
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
              className={activePanel === 'environment' ? 'active' : ''}
              onClick={() => togglePanel('environment')}
            >
              Environment
            </button>
            <button
              className={activePanel === 'recordings' ? 'active' : ''}
              onClick={() => togglePanel('recordings')}
            >
              Recordings ({recordings.length})
            </button>
            <button
              className={activePanel === 'ml' ? 'active' : ''}
              onClick={() => {
                togglePanel('ml');
                if (onFetchMLStatus) onFetchMLStatus();
              }}
            >
              ML/AI
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

          {/* Environment Content */}
          {activePanel === 'environment' && (
            <div className="advanced-content">
              <p className="panel-desc">Configure wind and atmospheric drag effects</p>

              <div className="env-controls">
                <div className="env-row">
                  <label>Wind Speed: {windSpeed} m/s</label>
                  <input
                    type="range"
                    min="0"
                    max="50"
                    step="1"
                    value={windSpeed}
                    onChange={(e) => setWindSpeed(parseInt(e.target.value))}
                    disabled={isRunning}
                  />
                </div>

                <div className="env-row">
                  <label>Wind Direction: {windDirection}°</label>
                  <input
                    type="range"
                    min="0"
                    max="360"
                    step="15"
                    value={windDirection}
                    onChange={(e) => setWindDirection(parseInt(e.target.value))}
                    disabled={isRunning}
                  />
                  <span className="wind-compass">
                    {windDirection === 0 ? 'N' : windDirection === 90 ? 'E' : windDirection === 180 ? 'S' : windDirection === 270 ? 'W' : `${windDirection}°`}
                  </span>
                </div>

                <div className="env-row">
                  <label>Wind Gusts: {windGusts} m/s</label>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="1"
                    value={windGusts}
                    onChange={(e) => setWindGusts(parseInt(e.target.value))}
                    disabled={isRunning}
                  />
                </div>

                <div className="env-row checkbox-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={enableDrag}
                      onChange={(e) => setEnableDrag(e.target.checked)}
                      disabled={isRunning}
                    />
                    Enable Atmospheric Drag
                  </label>
                </div>

                <div className="env-row checkbox-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={enableCooperative}
                      onChange={(e) => setEnableCooperative(e.target.checked)}
                      disabled={isRunning}
                    />
                    Enable Cooperative Engagement
                  </label>
                </div>
              </div>

              {/* Current environment state display */}
              {environmentState && environmentState.enabled && (
                <div className="env-state">
                  <div className="env-state-title">Current Wind</div>
                  <div className="env-state-row">
                    <span>X: {environmentState.current_wind?.x.toFixed(1) || 0} m/s</span>
                    <span>Y: {environmentState.current_wind?.y.toFixed(1) || 0} m/s</span>
                  </div>
                </div>
              )}

              {/* Cooperative engagement controls */}
              {enableCooperative && isRunning && cooperativeState?.enabled && (
                <div className="env-state">
                  <div className="env-state-title">Cooperative Engagement</div>
                  <div className="env-state-row">
                    <span>Zones: {cooperativeState?.engagement_zones?.length || 0}</span>
                    <span>Handoffs: {cooperativeState?.pending_handoffs?.length || 0}</span>
                  </div>
                  <button
                    className="btn-action"
                    style={{ marginTop: '8px' }}
                    onClick={() => {
                      if (onCreateEngagementZone) {
                        onCreateEngagementZone({
                          name: `Zone ${(cooperativeState?.engagement_zones?.length || 0) + 1}`,
                          center_x: 1500 + Math.random() * 500,
                          center_y: 0,
                          center_z: 600,
                          width: 800,
                          depth: 800,
                          height: 400,
                          rotation: 0,
                          priority: 1,
                          color: ['#00ff00', '#00ffff', '#ff00ff', '#ffff00'][
                            (cooperativeState?.engagement_zones?.length || 0) % 4
                          ],
                        });
                      }
                    }}
                  >
                    + Add Engagement Zone
                  </button>
                </div>
              )}

              {/* Help text when cooperative enabled but not running */}
              {enableCooperative && !isRunning && (
                <div className="env-state">
                  <div className="env-state-title">Cooperative Mode</div>
                  <p className="panel-desc" style={{ margin: '4px 0' }}>
                    Start a run to create engagement zones
                  </p>
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

          {/* ML/AI Content */}
          {activePanel === 'ml' && (
            <div className="advanced-content">
              <p className="panel-desc">Neural network threat assessment and RL guidance policies</p>

              <div className="ml-status">
                <div className="ml-status-row">
                  <span className="hud-label">ONNX Runtime</span>
                  <span className={`hud-value ${mlStatus?.onnx_available ? 'result-intercept' : 'result-missed'}`}>
                    {mlStatus?.onnx_available ? 'AVAILABLE' : 'NOT INSTALLED'}
                  </span>
                </div>

                {!mlStatus?.onnx_available && (
                  <div className="ml-install-hint">
                    <code>pip install onnxruntime</code>
                  </div>
                )}

                <div className="ml-section">
                  <div className="ml-section-title">Threat Models</div>
                  {mlStatus?.models?.threat_models && mlStatus.models.threat_models.length > 0 ? (
                    mlStatus.models.threat_models.map((model) => (
                      <div key={model.model_id} className="ml-model-row">
                        <span className="model-id">{model.model_id}</span>
                        <span className={`model-status ${model.active ? 'active' : ''}`}>
                          {model.active ? 'ACTIVE' : model.loaded ? 'LOADED' : 'UNLOADED'}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="ml-no-models">No models loaded</div>
                  )}
                </div>

                <div className="ml-section">
                  <div className="ml-section-title">Guidance Models</div>
                  {mlStatus?.models?.guidance_models && mlStatus.models.guidance_models.length > 0 ? (
                    mlStatus.models.guidance_models.map((model) => (
                      <div key={model.model_id} className="ml-model-row">
                        <span className="model-id">{model.model_id}</span>
                        <span className={`model-status ${model.active ? 'active' : ''}`}>
                          {model.active ? 'ACTIVE' : model.loaded ? 'LOADED' : 'UNLOADED'}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="ml-no-models">No models loaded</div>
                  )}
                </div>

                <div className="ml-info">
                  <p>Load ONNX models via API:</p>
                  <code>POST /ml/models/load</code>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}
