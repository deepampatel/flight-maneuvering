/**
 * MissionStatusHUD — Compact status panels
 *
 * Bottom-screen HUD showing mission status, target/interceptor
 * telemetry, threat assessment, WTA assignments, kill summary,
 * and replay controls.
 */

import type {
  SimStateEvent,
  InterceptGeometry,
  ThreatAssessment,
  AssignmentResult,
  ReplayState,
} from '../types';

interface MissionStatusHUDProps {
  state: SimStateEvent | null;
  interceptGeometry: InterceptGeometry[] | null;
  threatAssessment: ThreatAssessment[] | null;
  assignments: AssignmentResult | null;
  replayState: ReplayState | null;
  onPauseReplay: () => void;
  onResumeReplay: () => void;
  onStopReplay: () => void;
}

export function MissionStatusHUD({
  state,
  interceptGeometry,
  threatAssessment,
  assignments,
  replayState,
  onPauseReplay,
  onResumeReplay,
  onStopReplay,
}: MissionStatusHUDProps) {
  const targets = state?.entities.filter((e) => e.type === 'target') || [];
  const target = targets[0];
  const interceptors = state?.entities.filter((e) => e.type === 'interceptor') || [];

  return (
    <div className="telemetry-hud">
      {/* Mission status */}
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

      {/* Target */}
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

      {/* Interceptors */}
      {interceptors.slice(0, 4).map((int) => {
        const assignment = assignments?.assignments.find(a => a.interceptor_id === int.id);
        const assignedTargetId = assignment?.target_id;
        const geom = interceptGeometry?.find(
          g => g.interceptor_id === int.id && (assignedTargetId ? g.target_id === assignedTargetId : true)
        );
        const hasHit = state?.intercepted_pairs?.some(pair => pair[0] === int.id);

        return (
          <div key={int.id} className={`hud-panel interceptor-panel ${hasHit ? 'interceptor-hit' : ''}`}>
            <div className="hud-title interceptor">
              {int.id}
              {assignedTargetId && <span className="assigned-target">&rarr;{assignedTargetId}</span>}
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

      {/* Threat panel */}
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

      {/* WTA assignments */}
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

      {/* Multi-target kill summary */}
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

      {/* Replay controls */}
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
  );
}
