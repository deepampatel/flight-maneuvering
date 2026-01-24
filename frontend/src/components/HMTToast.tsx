/**
 * HMT Toast - Floating notification for Human-in-Loop pending actions
 *
 * Shows a compact, always-visible toast when there are actions awaiting approval.
 * Allows quick approve/reject without opening the Advanced panel.
 */

import { useEffect, useState } from 'react';
import type { PendingAction } from '../types';

interface HMTToastProps {
  pendingActions: PendingAction[];
  onApprove: (actionId: string) => void;
  onReject: (actionId: string) => void;
  enabled: boolean;
}

export function HMTToast({ pendingActions, onApprove, onReject, enabled }: HMTToastProps) {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState(false);

  // Reset dismissed when new actions come in
  useEffect(() => {
    const currentIds = new Set(pendingActions.map(a => a.action_id));
    setDismissed(prev => {
      const newDismissed = new Set<string>();
      prev.forEach(id => {
        if (currentIds.has(id)) newDismissed.add(id);
      });
      return newDismissed;
    });
  }, [pendingActions]);

  // Filter out dismissed actions
  const visibleActions = pendingActions.filter(a => !dismissed.has(a.action_id));

  if (!enabled || visibleActions.length === 0) {
    return null;
  }

  const handleApprove = (actionId: string) => {
    onApprove(actionId);
    setDismissed(prev => new Set(prev).add(actionId));
  };

  const handleReject = (actionId: string) => {
    onReject(actionId);
    setDismissed(prev => new Set(prev).add(actionId));
  };

  const handleApproveAll = () => {
    visibleActions.forEach(a => {
      onApprove(a.action_id);
      setDismissed(prev => new Set(prev).add(a.action_id));
    });
  };

  // Show first action prominently, others in expanded view
  const primaryAction = visibleActions[0];
  const otherActions = visibleActions.slice(1);

  return (
    <div className="hmt-toast">
      <div className="hmt-toast-header">
        <span className="hmt-toast-icon">⚡</span>
        <span className="hmt-toast-title">ACTION REQUIRED</span>
        <span className="hmt-toast-count">{visibleActions.length}</span>
      </div>

      {/* Primary Action */}
      <div className="hmt-toast-action primary">
        <div className="hmt-action-info">
          <span className="hmt-action-type">{primaryAction.action_type.toUpperCase()}</span>
          <span className="hmt-action-entities">
            {primaryAction.entity_id} → {primaryAction.target_id}
          </span>
          <span className="hmt-action-confidence">
            {(primaryAction.confidence * 100).toFixed(0)}%
          </span>
          <div className="hmt-action-timer">
            <div
              className="hmt-timer-bar"
              style={{
                width: `${Math.max(0, (primaryAction.time_remaining / 5) * 100)}%`,
                backgroundColor: primaryAction.time_remaining < 2 ? '#ff3b30' : '#00d4aa'
              }}
            />
          </div>
        </div>
        <div className="hmt-action-buttons">
          <button
            className="hmt-btn-approve"
            onClick={() => handleApprove(primaryAction.action_id)}
            title="Approve (Enter)"
          >
            ✓
          </button>
          <button
            className="hmt-btn-reject"
            onClick={() => handleReject(primaryAction.action_id)}
            title="Reject (Esc)"
          >
            ✗
          </button>
        </div>
      </div>

      {/* Other Actions (collapsed by default) */}
      {otherActions.length > 0 && (
        <>
          <button
            className="hmt-toast-expand"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? '▼' : '▶'} {otherActions.length} more action{otherActions.length > 1 ? 's' : ''}
          </button>

          {expanded && (
            <div className="hmt-toast-expanded">
              {otherActions.map(action => (
                <div key={action.action_id} className="hmt-toast-action secondary">
                  <div className="hmt-action-info">
                    <span className="hmt-action-type">{action.action_type.toUpperCase()}</span>
                    <span className="hmt-action-entities">
                      {action.entity_id} → {action.target_id}
                    </span>
                  </div>
                  <div className="hmt-action-buttons">
                    <button
                      className="hmt-btn-approve small"
                      onClick={() => handleApprove(action.action_id)}
                    >
                      ✓
                    </button>
                    <button
                      className="hmt-btn-reject small"
                      onClick={() => handleReject(action.action_id)}
                    >
                      ✗
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Approve All button */}
      {visibleActions.length > 1 && (
        <button className="hmt-approve-all" onClick={handleApproveAll}>
          APPROVE ALL ({visibleActions.length})
        </button>
      )}
    </div>
  );
}
