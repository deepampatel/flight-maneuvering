/**
 * HMTTab — Human-Machine Teaming
 *
 * Configure authority levels, view pending actions requiring human
 * approval, and monitor trust/workload metrics.
 */

import type {
  HMTStatus,
  HMTConfig,
  PendingAction,
  AuthorityLevel,
  AuthorityLevelInfo,
} from '../../types';

interface HMTTabProps {
  isRunning: boolean;
  enableHmt: boolean;
  hmtAuthorityLevel: AuthorityLevel;
  onEnableHmt: (v: boolean) => void;
  onHmtAuthorityLevel: (v: AuthorityLevel) => void;
  // Live state
  hmtStatus?: HMTStatus | null;
  authorityLevels?: AuthorityLevelInfo[];
  pendingActions?: PendingAction[];
  onSetAuthorityLevel?: (level: AuthorityLevel) => void;
  onConfigureHMT?: (config: Partial<HMTConfig>) => void;
  onApproveAction?: (actionId: string, reason?: string) => void;
  onRejectAction?: (actionId: string, reason?: string) => void;
}

export function HMTTab({
  isRunning,
  enableHmt,
  hmtAuthorityLevel,
  onEnableHmt,
  onHmtAuthorityLevel,
  hmtStatus,
  authorityLevels,
  pendingActions,
  onSetAuthorityLevel,
  onApproveAction,
  onRejectAction,
}: HMTTabProps) {
  void onSetAuthorityLevel; // Used via onHmtAuthorityLevel which wraps it
  return (
    <div className="advanced-content">
      <p className="panel-desc">Human-Machine Teaming: Control automation authority</p>

      <div className="env-controls">
        <div className="env-row checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={enableHmt}
              onChange={(e) => onEnableHmt(e.target.checked)}
              disabled={isRunning}
            />
            Enable Human-Machine Teaming
          </label>
        </div>

        {enableHmt && (
          <div className="env-row">
            <label>Authority Level</label>
            <select
              value={hmtAuthorityLevel}
              onChange={(e) => {
                const level = e.target.value as AuthorityLevel;
                onHmtAuthorityLevel(level);
              }}
            >
              {authorityLevels && authorityLevels.length > 0 ? (
                authorityLevels.map((a) => (
                  <option key={a.id} value={a.id} title={a.description}>
                    {a.name}
                  </option>
                ))
              ) : (
                <>
                  <option value="full_auto">Full Auto</option>
                  <option value="human_on_loop">Human on Loop</option>
                  <option value="human_in_loop">Human in Loop</option>
                  <option value="manual">Manual</option>
                </>
              )}
            </select>
          </div>
        )}

        {/* HMT Status */}
        {isRunning && enableHmt && hmtStatus?.enabled && (
          <div className="env-state">
            <div className="env-state-title">HMT Metrics</div>
            <div className="env-state-row">
              <span>Authority: {hmtStatus.metrics?.authority_level || hmtAuthorityLevel}</span>
            </div>
            <div className="env-state-row">
              <span>Workload: {hmtStatus.metrics?.workload.actions_per_minute?.toFixed(1) || 0}/min</span>
            </div>
            <div className="env-state-row">
              <span>Trust: {((hmtStatus.metrics?.trust.ai_accuracy || 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}

        {/* Pending Actions */}
        {isRunning && enableHmt && pendingActions && pendingActions.length > 0 && (
          <div className="env-state">
            <div className="env-state-title" style={{ color: '#ff9500' }}>
              Pending Actions ({pendingActions.length})
            </div>
            <div className="pending-actions-list">
              {pendingActions.slice(0, 5).map((action) => (
                <div key={action.action_id} className="pending-action">
                  <div className="action-info">
                    <span className="action-type">{action.action_type.toUpperCase()}</span>
                    <span className="action-entity">{action.entity_id}</span>
                    {action.target_id && (
                      <span className="action-target">&rarr; {action.target_id}</span>
                    )}
                    <span className="action-confidence">
                      {(action.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="action-timeout">
                      {action.time_remaining.toFixed(1)}s
                    </span>
                  </div>
                  <div className="action-buttons">
                    <button
                      className="btn-approve"
                      onClick={() => onApproveAction && onApproveAction(action.action_id)}
                    >
                      &#x2713;
                    </button>
                    <button
                      className="btn-reject"
                      onClick={() => onRejectAction && onRejectAction(action.action_id)}
                    >
                      &#x2717;
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Authority Level Descriptions */}
        {enableHmt && !isRunning && (
          <div className="env-state">
            <div className="env-state-title">Authority Levels</div>
            <div className="authority-descriptions">
              <div className={`auth-desc ${hmtAuthorityLevel === 'full_auto' ? 'active' : ''}`}>
                <strong>Full Auto:</strong> AI acts autonomously, human notified
              </div>
              <div className={`auth-desc ${hmtAuthorityLevel === 'human_on_loop' ? 'active' : ''}`}>
                <strong>Human on Loop:</strong> AI acts, human can override
              </div>
              <div className={`auth-desc ${hmtAuthorityLevel === 'human_in_loop' ? 'active' : ''}`}>
                <strong>Human in Loop:</strong> AI proposes, human approves
              </div>
              <div className={`auth-desc ${hmtAuthorityLevel === 'manual' ? 'active' : ''}`}>
                <strong>Manual:</strong> Human controls all actions
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
