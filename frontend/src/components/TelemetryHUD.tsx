/**
 * TelemetryHUD - Real-time flight data overlay on the 3D scene
 *
 * Displays selected entity telemetry (Mach, altitude, G-load, heading)
 * and engagement status (TTI, miss distance, collision course).
 * Positioned as an absolute overlay on the scene container.
 */

import { useMemo } from 'react';
import type { EntityState, SimStateEvent, InterceptGeometry } from '../types';

interface TelemetryHUDProps {
  state: SimStateEvent | null;
  selectedEntityId: string | null;
  interceptGeometry: InterceptGeometry | null;
}

function formatHeading(vx: number, vy: number): string {
  let deg = Math.atan2(vx, vy) * (180 / Math.PI);
  if (deg < 0) deg += 360;
  return `${deg.toFixed(0).padStart(3, '0')}°`;
}

function headingToCompass(vx: number, vy: number): string {
  let deg = Math.atan2(vx, vy) * (180 / Math.PI);
  if (deg < 0) deg += 360;
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return dirs[Math.round(deg / 45) % 8];
}

export function TelemetryHUD({ state, selectedEntityId, interceptGeometry }: TelemetryHUDProps) {
  const selectedEntity = useMemo(() => {
    if (!state?.entities || !selectedEntityId) return null;
    return state.entities.find((e: EntityState) => e.id === selectedEntityId) || null;
  }, [state?.entities, selectedEntityId]);

  const isRunning = state?.status === 'running';

  if (!isRunning) return null;

  return (
    <div className="telemetry-overlay">
      {/* Selected Entity Card - Top Left */}
      {selectedEntity && (
        <div className="telem-card telem-entity">
          <div className="telem-card-header">
            <span className={`telem-entity-type ${selectedEntity.type}`}>
              {selectedEntity.type === 'interceptor' ? '▲' : '●'}
            </span>
            <span className="telem-entity-id">{selectedEntity.id}</span>
            <span className={`telem-entity-badge ${selectedEntity.type}`}>
              {selectedEntity.type.toUpperCase()}
            </span>
          </div>
          <div className="telem-grid">
            <div className="telem-metric">
              <span className="telem-label">MACH</span>
              <span className="telem-value telem-value-lg">
                {(selectedEntity.speed / 343).toFixed(2)}
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">ALT</span>
              <span className="telem-value">
                {(selectedEntity.position.z / 1000).toFixed(1)}km
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">G-LOAD</span>
              <span className={`telem-value ${
                Math.sqrt(
                  selectedEntity.acceleration.x ** 2 +
                  selectedEntity.acceleration.y ** 2 +
                  selectedEntity.acceleration.z ** 2
                ) / 9.81 > 5 ? 'telem-value-warn' : ''
              }`}>
                {(Math.sqrt(
                  selectedEntity.acceleration.x ** 2 +
                  selectedEntity.acceleration.y ** 2 +
                  selectedEntity.acceleration.z ** 2
                ) / 9.81).toFixed(1)}G
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">HDG</span>
              <span className="telem-value">
                {formatHeading(selectedEntity.velocity.x, selectedEntity.velocity.y)}
                <span className="telem-value-sub">
                  {headingToCompass(selectedEntity.velocity.x, selectedEntity.velocity.y)}
                </span>
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">SPD</span>
              <span className="telem-value">{selectedEntity.speed.toFixed(0)} m/s</span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">POS</span>
              <span className="telem-value telem-value-sm">
                {(selectedEntity.position.x / 1000).toFixed(1)}, {(selectedEntity.position.y / 1000).toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Engagement Status - Top Right */}
      {interceptGeometry && (
        <div className="telem-card telem-engagement">
          <div className="telem-card-header">
            <span className="telem-card-title">ENGAGEMENT</span>
            {interceptGeometry.collision_course && (
              <span className="telem-collision-badge">COLLISION COURSE</span>
            )}
          </div>
          <div className="telem-grid">
            <div className="telem-metric">
              <span className="telem-label">TTI</span>
              <span className={`telem-value telem-value-lg ${
                interceptGeometry.time_to_intercept < 3 ? 'telem-value-critical' :
                interceptGeometry.time_to_intercept < 8 ? 'telem-value-warn' : ''
              }`}>
                {interceptGeometry.time_to_intercept.toFixed(1)}s
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">RANGE</span>
              <span className="telem-value">
                {(interceptGeometry.los_range / 1000).toFixed(2)}km
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">MISS DIST</span>
              <span className={`telem-value ${
                interceptGeometry.predicted_miss_distance < 200 ? 'telem-value-good' : 'telem-value-warn'
              }`}>
                {interceptGeometry.predicted_miss_distance.toFixed(0)}m
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">CLOSING</span>
              <span className="telem-value">
                {interceptGeometry.closing_velocity.toFixed(0)} m/s
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">ASPECT</span>
              <span className="telem-value">
                {(interceptGeometry.aspect_angle * 180 / Math.PI).toFixed(0)}°
              </span>
            </div>
            <div className="telem-metric">
              <span className="telem-label">LEAD</span>
              <span className="telem-value">
                {(interceptGeometry.lead_angle * 180 / Math.PI).toFixed(1)}°
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Bottom Center - Compact Status */}
      <div className="telem-status-bar">
        <span className="telem-status-item">
          <span className="telem-label">T</span>
          <span className="telem-value">{state?.sim_time?.toFixed(1)}s</span>
        </span>
        <span className="telem-status-divider" />
        <span className="telem-status-item">
          <span className="telem-label">MISS</span>
          <span className={`telem-value ${
            state?.miss_distance && state.miss_distance < 200 ? 'telem-value-good' : ''
          }`}>
            {state?.miss_distance ? `${state.miss_distance.toFixed(0)}m` : '---'}
          </span>
        </span>
        <span className="telem-status-divider" />
        <span className="telem-status-item">
          <span className="telem-label">RESULT</span>
          <span className={`telem-value telem-result-${state?.result || 'pending'}`}>
            {(state?.result || 'TRACKING').toUpperCase()}
          </span>
        </span>
      </div>
    </div>
  );
}
