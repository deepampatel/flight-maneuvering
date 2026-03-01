/**
 * BatteryStatusPanel — Compact ammo/engagement status for Iron Dome batteries
 */

import type { BatteryState } from '../types';

interface BatteryStatusPanelProps {
  batteries: BatteryState[];
}

export function BatteryStatusPanel({ batteries }: BatteryStatusPanelProps) {
  if (!batteries || batteries.length === 0) return null;

  return (
    <div className="battery-status-container">
      {batteries.map((battery) => {
        const ammoPercent = battery.missiles_total > 0
          ? battery.missiles_remaining / battery.missiles_total
          : 0;
        const statusColor = battery.status === 'operational' ? '#22c55e'
          : battery.status === 'degraded' ? '#f59e0b'
          : battery.status === 'winchester' ? '#ef4444'
          : '#6b7280';

        return (
          <div key={battery.id} className="hud-panel battery-panel">
            <div className="hud-title" style={{ color: statusColor }}>
              {battery.name.toUpperCase()}
            </div>
            <div className="hud-content">
              {/* Status */}
              <div className="hud-row">
                <span className="hud-label">STATUS</span>
                <span className="hud-value" style={{ color: statusColor }}>
                  {battery.status.toUpperCase()}
                </span>
              </div>

              {/* Ammo gauge */}
              <div className="hud-row">
                <span className="hud-label">AMMO</span>
                <span className="hud-value">
                  {battery.missiles_remaining}/{battery.missiles_total}
                </span>
              </div>
              <div className="battery-ammo-bar">
                <div
                  className="battery-ammo-fill"
                  style={{
                    width: `${ammoPercent * 100}%`,
                    backgroundColor: ammoPercent > 0.5 ? '#22c55e'
                      : ammoPercent > 0.25 ? '#f59e0b'
                      : '#ef4444',
                  }}
                />
              </div>

              {/* Active engagements */}
              <div className="hud-row">
                <span className="hud-label">ACTIVE</span>
                <span className="hud-value">
                  {battery.active_engagements}/{battery.max_simultaneous}
                </span>
              </div>

              {/* Tracked threats */}
              <div className="hud-row">
                <span className="hud-label">TRACKS</span>
                <span className="hud-value">{battery.tracked_threats}</span>
              </div>

              {/* Per-launcher status (compact dots) */}
              <div className="battery-launchers">
                {battery.launchers.map((launcher) => {
                  const lPercent = launcher.missiles_total > 0
                    ? launcher.missiles_remaining / launcher.missiles_total
                    : 0;
                  return (
                    <div
                      key={launcher.id}
                      className="battery-launcher-dot"
                      title={`${launcher.id}: ${launcher.missiles_remaining}/${launcher.missiles_total}`}
                      style={{
                        backgroundColor: lPercent > 0.5 ? '#22c55e'
                          : lPercent > 0 ? '#f59e0b'
                          : '#ef4444',
                        opacity: launcher.ready ? 1 : 0.4,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
