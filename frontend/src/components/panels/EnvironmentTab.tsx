/**
 * EnvironmentTab — Wind, drag, sensor, cooperative engagement config
 *
 * Configures atmospheric conditions and cooperative engagement zones.
 */

import type {
  EnvironmentState,
  CooperativeState,
  EngagementZoneCreateRequest,
} from '../../types';

interface EnvironmentTabProps {
  isRunning: boolean;
  // Wind & drag
  windSpeed: number;
  windDirection: number;
  windGusts: number;
  enableDrag: boolean;
  onWindSpeed: (v: number) => void;
  onWindDirection: (v: number) => void;
  onWindGusts: (v: number) => void;
  onEnableDrag: (v: boolean) => void;
  // Cooperative
  enableCooperative: boolean;
  onEnableCooperative: (v: boolean) => void;
  // Live state
  environmentState: EnvironmentState | null;
  cooperativeState?: CooperativeState | null;
  onCreateEngagementZone?: (zone: EngagementZoneCreateRequest) => void;
}

export function EnvironmentTab({
  isRunning,
  windSpeed,
  windDirection,
  windGusts,
  enableDrag,
  onWindSpeed,
  onWindDirection,
  onWindGusts,
  onEnableDrag,
  enableCooperative,
  onEnableCooperative,
  environmentState,
  cooperativeState,
  onCreateEngagementZone,
}: EnvironmentTabProps) {
  return (
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
            onChange={(e) => onWindSpeed(parseInt(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="env-row">
          <label>Wind Direction: {windDirection}&deg;</label>
          <input
            type="range"
            min="0"
            max="360"
            step="15"
            value={windDirection}
            onChange={(e) => onWindDirection(parseInt(e.target.value))}
            disabled={isRunning}
          />
          <span className="wind-compass">
            {windDirection === 0 ? 'N' : windDirection === 90 ? 'E' : windDirection === 180 ? 'S' : windDirection === 270 ? 'W' : `${windDirection}\u00b0`}
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
            onChange={(e) => onWindGusts(parseInt(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="env-row checkbox-row">
          <label>
            <input
              type="checkbox"
              checked={enableDrag}
              onChange={(e) => onEnableDrag(e.target.checked)}
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
              onChange={(e) => onEnableCooperative(e.target.checked)}
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
  );
}
