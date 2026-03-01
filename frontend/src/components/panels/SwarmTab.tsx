/**
 * SwarmTab — Swarm tactics and formation control
 *
 * Configure interceptor formations, spacing, and view real-time
 * swarm status during simulation.
 */

import type {
  SwarmStatus,
  SwarmConfig,
  FormationInfo,
  FormationType,
} from '../../types';

interface SwarmTabProps {
  isRunning: boolean;
  enableSwarm: boolean;
  swarmFormation: FormationType;
  swarmSpacing: number;
  onEnableSwarm: (v: boolean) => void;
  onSwarmFormation: (v: FormationType) => void;
  onSwarmSpacing: (v: number) => void;
  // Live state
  swarmStatus?: SwarmStatus | null;
  formationTypes?: FormationInfo[];
  onConfigureSwarm?: (config: Partial<SwarmConfig>) => void;
  onSetSwarmFormation?: (formation: FormationType) => void;
}

export function SwarmTab({
  isRunning,
  enableSwarm,
  swarmFormation,
  swarmSpacing,
  onEnableSwarm,
  onSwarmFormation,
  onSwarmSpacing,
  swarmStatus,
  formationTypes,
  onConfigureSwarm,
  onSetSwarmFormation,
}: SwarmTabProps) {
  return (
    <div className="advanced-content">
      <p className="panel-desc">Configure swarm tactics and formations</p>

      <div className="env-controls">
        <div className="env-row checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={enableSwarm}
              onChange={(e) => onEnableSwarm(e.target.checked)}
              disabled={isRunning}
            />
            Enable Swarm Tactics
          </label>
        </div>

        {enableSwarm && (
          <>
            <div className="env-row">
              <label>Formation</label>
              <select
                value={swarmFormation}
                onChange={(e) => {
                  onSwarmFormation(e.target.value as FormationType);
                  if (isRunning && onSetSwarmFormation) {
                    onSetSwarmFormation(e.target.value as FormationType);
                  }
                }}
              >
                {formationTypes && formationTypes.length > 0 ? (
                  formationTypes.map((f) => (
                    <option key={f.id} value={f.id} title={f.description}>
                      {f.name}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="line_abreast">Line Abreast</option>
                    <option value="echelon_right">Echelon Right</option>
                    <option value="echelon_left">Echelon Left</option>
                    <option value="v_formation">V-Formation</option>
                    <option value="wedge">Wedge</option>
                    <option value="trail">Trail</option>
                    <option value="diamond">Diamond</option>
                    <option value="swarm">Free Swarm</option>
                  </>
                )}
              </select>
            </div>

            <div className="env-row">
              <label>Spacing: {swarmSpacing}m</label>
              <input
                type="range"
                min="50"
                max="500"
                step="25"
                value={swarmSpacing}
                onChange={(e) => {
                  const value = parseInt(e.target.value);
                  onSwarmSpacing(value);
                  if (isRunning && onConfigureSwarm) {
                    onConfigureSwarm({ spacing: value });
                  }
                }}
              />
            </div>
          </>
        )}

        {/* Swarm status during simulation */}
        {isRunning && enableSwarm && swarmStatus?.enabled && (
          <div className="env-state">
            <div className="env-state-title">Swarm Status</div>
            <div className="env-state-row">
              <span>Leader: {swarmStatus.state?.leader_id || 'None'}</span>
            </div>
            <div className="env-state-row">
              <span>Formation: {swarmStatus.state?.formation || swarmFormation}</span>
            </div>
            <div className="env-state-row">
              <span>Error: {swarmStatus.state?.formation_error?.toFixed(1) || 0}m</span>
              <span>Cohesion: {((swarmStatus.state?.cohesion_metric || 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}

        {/* Help text when swarm enabled but not running */}
        {enableSwarm && !isRunning && (
          <div className="env-state">
            <div className="env-state-title">Swarm Mode</div>
            <p className="panel-desc" style={{ margin: '4px 0' }}>
              Interceptors will fly in {swarmFormation.replace('_', ' ')} formation
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
