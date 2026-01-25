/**
 * MissionPlannerPanel - Clean Sidebar for Scenario Building
 *
 * World-class UX for mission planning:
 * 1. Entity list with expandable cards
 * 2. Simple speed + heading controls (intuitive velocity)
 * 3. Clean numeric inputs for position
 * 4. Drag-to-reorder entities
 * 5. Quick presets for common setups
 */

import { useState, useCallback } from 'react';
import type { PlannedEntity, PlannedZone, PlacementMode } from './MissionPlanner';

interface MissionPlannerPanelProps {
  mode: PlacementMode;
  onSetMode: (mode: PlacementMode) => void;
  plannedEntities: PlannedEntity[];
  plannedZones: PlannedZone[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onUpdateEntity: (id: string, updates: Partial<PlannedEntity>) => void;
  onRemoveEntity: (id: string) => void;
  onUpdateZone: (id: string, updates: Partial<PlannedZone>) => void;
  onRemoveZone: (id: string) => void;
  onClearAll: () => void;
  showGrid: boolean;
  onToggleGrid: () => void;
  snapToGrid: boolean;
  onToggleSnap: () => void;
  isSimRunning: boolean;
}

// Convert velocity to speed + heading
function velocityToSpeedHeading(vx: number, vy: number): { speed: number; heading: number } {
  const speed = Math.sqrt(vx * vx + vy * vy);
  // Heading: 0 = North (+Y), 90 = East (+X), 180 = South (-Y), 270 = West (-X)
  let heading = Math.atan2(vx, vy) * (180 / Math.PI);
  if (heading < 0) heading += 360;
  return { speed: Math.round(speed), heading: Math.round(heading) };
}

// Convert speed + heading to velocity
function speedHeadingToVelocity(speed: number, heading: number): { x: number; y: number } {
  const rad = heading * (Math.PI / 180);
  return {
    x: speed * Math.sin(rad),  // East component
    y: speed * Math.cos(rad),  // North component
  };
}

// Format heading as compass direction
function headingToCompass(heading: number): string {
  const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const idx = Math.round(heading / 45) % 8;
  return dirs[idx];
}

export function MissionPlannerPanel({
  mode,
  onSetMode,
  plannedEntities,
  plannedZones,
  selectedEntityId,
  onSelectEntity,
  onUpdateEntity,
  onRemoveEntity,
  onUpdateZone,
  onRemoveZone,
  onClearAll,
  showGrid,
  onToggleGrid,
  snapToGrid,
  onToggleSnap,
  isSimRunning,
}: MissionPlannerPanelProps) {
  const [expandedEntity, setExpandedEntity] = useState<string | null>(null);

  const interceptors = plannedEntities.filter(e => e.type === 'interceptor');
  const targets = plannedEntities.filter(e => e.type === 'target');
  const launchers = plannedEntities.filter(e => e.type === 'launcher');

  const handleEntityClick = useCallback((id: string) => {
    onSelectEntity(id);
    setExpandedEntity(prev => prev === id ? null : id);
  }, [onSelectEntity]);

  const handlePositionChange = useCallback((id: string, entity: PlannedEntity, axis: 'x' | 'y' | 'z', value: number) => {
    onUpdateEntity(id, {
      position: { ...entity.position, [axis]: value }
    });
  }, [onUpdateEntity]);

  const handleVelocityChange = useCallback((id: string, entity: PlannedEntity, field: 'speed' | 'heading', value: number) => {
    const current = velocityToSpeedHeading(entity.velocity.x, entity.velocity.y);
    const newSpeed = field === 'speed' ? value : current.speed;
    const newHeading = field === 'heading' ? value : current.heading;
    const vel = speedHeadingToVelocity(newSpeed, newHeading);
    onUpdateEntity(id, {
      velocity: { x: vel.x, y: vel.y, z: entity.velocity.z }
    });
  }, [onUpdateEntity]);

  const renderEntityCard = (entity: PlannedEntity) => {
    const isSelected = selectedEntityId === entity.id;
    const isExpanded = expandedEntity === entity.id;
    const { speed, heading } = velocityToSpeedHeading(entity.velocity.x, entity.velocity.y);
    const isLauncher = entity.type === 'launcher';

    return (
      <div
        key={entity.id}
        className={`mp-entity-card ${entity.type} ${isSelected ? 'selected' : ''} ${isExpanded ? 'expanded' : ''}`}
        onClick={() => handleEntityClick(entity.id)}
      >
        {/* Header Row */}
        <div className="mp-entity-header">
          <div className="mp-entity-icon">
            {entity.type === 'interceptor' && '▲'}
            {entity.type === 'target' && '●'}
            {entity.type === 'launcher' && '◬'}
          </div>
          <div className="mp-entity-info">
            <span className="mp-entity-id">{entity.id}</span>
            {!isLauncher && (
              <span className="mp-entity-velocity">
                {speed} m/s {headingToCompass(heading)}
              </span>
            )}
            {isLauncher && entity.launcherConfig && (
              <span className="mp-entity-velocity">
                {entity.launcherConfig.numMissiles} missiles
              </span>
            )}
          </div>
          <button
            className="mp-entity-delete"
            onClick={(e) => {
              e.stopPropagation();
              onRemoveEntity(entity.id);
            }}
            title="Delete"
          >
            ×
          </button>
        </div>

        {/* Expanded Controls */}
        {isExpanded && (
          <div className="mp-entity-controls" onClick={e => e.stopPropagation()}>
            {/* Position */}
            <div className="mp-control-group">
              <label className="mp-control-label">Position</label>
              <div className="mp-position-inputs">
                <div className="mp-input-row">
                  <span className="mp-input-label">X</span>
                  <input
                    type="number"
                    value={Math.round(entity.position.x)}
                    onChange={(e) => handlePositionChange(entity.id, entity, 'x', parseFloat(e.target.value) || 0)}
                    className="mp-input"
                  />
                  <span className="mp-input-unit">m</span>
                </div>
                <div className="mp-input-row">
                  <span className="mp-input-label">Y</span>
                  <input
                    type="number"
                    value={Math.round(entity.position.y)}
                    onChange={(e) => handlePositionChange(entity.id, entity, 'y', parseFloat(e.target.value) || 0)}
                    className="mp-input"
                  />
                  <span className="mp-input-unit">m</span>
                </div>
                <div className="mp-input-row">
                  <span className="mp-input-label">Alt</span>
                  <input
                    type="number"
                    value={Math.round(entity.position.z)}
                    onChange={(e) => handlePositionChange(entity.id, entity, 'z', parseFloat(e.target.value) || 0)}
                    className="mp-input"
                  />
                  <span className="mp-input-unit">m</span>
                </div>
              </div>
            </div>

            {/* Velocity (not for launchers) */}
            {!isLauncher && (
              <div className="mp-control-group">
                <label className="mp-control-label">Velocity</label>
                <div className="mp-velocity-controls">
                  <div className="mp-slider-row">
                    <span className="mp-slider-label">Speed</span>
                    <input
                      type="range"
                      min="0"
                      max="400"
                      value={speed}
                      onChange={(e) => handleVelocityChange(entity.id, entity, 'speed', parseFloat(e.target.value))}
                      className="mp-slider"
                    />
                    <input
                      type="number"
                      value={speed}
                      onChange={(e) => handleVelocityChange(entity.id, entity, 'speed', parseFloat(e.target.value) || 0)}
                      className="mp-input small"
                      min="0"
                      max="400"
                    />
                    <span className="mp-input-unit">m/s</span>
                  </div>
                  <div className="mp-slider-row">
                    <span className="mp-slider-label">Heading</span>
                    <input
                      type="range"
                      min="0"
                      max="359"
                      value={heading}
                      onChange={(e) => handleVelocityChange(entity.id, entity, 'heading', parseFloat(e.target.value))}
                      className="mp-slider"
                    />
                    <input
                      type="number"
                      value={heading}
                      onChange={(e) => handleVelocityChange(entity.id, entity, 'heading', parseFloat(e.target.value) || 0)}
                      className="mp-input small"
                      min="0"
                      max="359"
                    />
                    <span className="mp-input-unit">°{headingToCompass(heading)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Launcher Config */}
            {isLauncher && entity.launcherConfig && (
              <div className="mp-control-group">
                <label className="mp-control-label">Launcher Config</label>
                <div className="mp-position-inputs">
                  <div className="mp-input-row">
                    <span className="mp-input-label">Missiles</span>
                    <input
                      type="number"
                      value={entity.launcherConfig.numMissiles}
                      onChange={(e) => onUpdateEntity(entity.id, {
                        launcherConfig: { ...entity.launcherConfig!, numMissiles: parseInt(e.target.value) || 1 }
                      })}
                      className="mp-input"
                      min="1"
                      max="20"
                    />
                  </div>
                  <div className="mp-input-row">
                    <span className="mp-input-label">Range</span>
                    <input
                      type="number"
                      value={entity.launcherConfig.detectionRange}
                      onChange={(e) => onUpdateEntity(entity.id, {
                        launcherConfig: { ...entity.launcherConfig!, detectionRange: parseInt(e.target.value) || 1000 }
                      })}
                      className="mp-input"
                      step="500"
                    />
                    <span className="mp-input-unit">m</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderZoneCard = (zone: PlannedZone) => {
    const isSelected = selectedEntityId === zone.id;

    return (
      <div
        key={zone.id}
        className={`mp-entity-card zone ${isSelected ? 'selected' : ''}`}
        onClick={() => onSelectEntity(zone.id)}
      >
        <div className="mp-entity-header">
          <div className="mp-entity-icon">▢</div>
          <div className="mp-entity-info">
            <span className="mp-entity-id">{zone.name}</span>
            <span className="mp-entity-velocity">
              {Math.round(zone.dimensions.x)}×{Math.round(zone.dimensions.y)}m
            </span>
          </div>
          <button
            className="mp-entity-delete"
            onClick={(e) => {
              e.stopPropagation();
              onRemoveZone(zone.id);
            }}
            title="Delete"
          >
            ×
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className={`mission-planner-panel ${isSimRunning ? 'disabled' : ''}`}>
      {/* Header */}
      <div className="mp-header">
        <h3 className="mp-title">Mission Planner</h3>
        {(plannedEntities.length > 0 || plannedZones.length > 0) && (
          <button className="mp-clear-btn" onClick={onClearAll} title="Clear All">
            Clear
          </button>
        )}
      </div>

      {/* Mode Selector */}
      <div className="mp-modes">
        <button
          className={`mp-mode-btn ${mode === 'view' ? 'active' : ''}`}
          onClick={() => onSetMode('view')}
          title="Select & Edit"
        >
          <span className="mp-mode-icon">◆</span>
          <span>Select</span>
        </button>
        <button
          className={`mp-mode-btn interceptor ${mode === 'interceptor' ? 'active' : ''}`}
          onClick={() => onSetMode('interceptor')}
          title="Add Interceptor"
        >
          <span className="mp-mode-icon">▲</span>
          <span>Interceptor</span>
        </button>
        <button
          className={`mp-mode-btn target ${mode === 'target' ? 'active' : ''}`}
          onClick={() => onSetMode('target')}
          title="Add Target"
        >
          <span className="mp-mode-icon">●</span>
          <span>Target</span>
        </button>
        <button
          className={`mp-mode-btn launcher ${mode === 'launcher' ? 'active' : ''}`}
          onClick={() => onSetMode('launcher')}
          title="Add Launcher"
        >
          <span className="mp-mode-icon">◬</span>
          <span>Launcher</span>
        </button>
        <button
          className={`mp-mode-btn zone ${mode === 'zone' ? 'active' : ''}`}
          onClick={() => onSetMode('zone')}
          title="Draw Zone"
        >
          <span className="mp-mode-icon">▢</span>
          <span>Zone</span>
        </button>
      </div>

      {/* Helpers */}
      <div className="mp-helpers">
        <button
          className={`mp-helper-btn ${showGrid ? 'active' : ''}`}
          onClick={onToggleGrid}
        >
          Grid
        </button>
        <button
          className={`mp-helper-btn ${snapToGrid ? 'active' : ''}`}
          onClick={onToggleSnap}
        >
          Snap
        </button>
      </div>

      {/* Entity Lists */}
      <div className="mp-entity-lists">
        {/* Interceptors */}
        {interceptors.length > 0 && (
          <div className="mp-entity-section">
            <div className="mp-section-header interceptor">
              <span>Interceptors</span>
              <span className="mp-section-count">{interceptors.length}</span>
            </div>
            {interceptors.map(renderEntityCard)}
          </div>
        )}

        {/* Targets */}
        {targets.length > 0 && (
          <div className="mp-entity-section">
            <div className="mp-section-header target">
              <span>Targets</span>
              <span className="mp-section-count">{targets.length}</span>
            </div>
            {targets.map(renderEntityCard)}
          </div>
        )}

        {/* Launchers */}
        {launchers.length > 0 && (
          <div className="mp-entity-section">
            <div className="mp-section-header launcher">
              <span>Launchers</span>
              <span className="mp-section-count">{launchers.length}</span>
            </div>
            {launchers.map(renderEntityCard)}
          </div>
        )}

        {/* Zones */}
        {plannedZones.length > 0 && (
          <div className="mp-entity-section">
            <div className="mp-section-header zone">
              <span>Zones</span>
              <span className="mp-section-count">{plannedZones.length}</span>
            </div>
            {plannedZones.map(renderZoneCard)}
          </div>
        )}

        {/* Empty State */}
        {plannedEntities.length === 0 && plannedZones.length === 0 && (
          <div className="mp-empty-state">
            <p>Click in the 3D view to place entities</p>
            <p className="mp-hint">
              {mode === 'view' && 'Select a mode above to start'}
              {mode === 'interceptor' && 'Click & drag to place interceptor'}
              {mode === 'target' && 'Click & drag to place target'}
              {mode === 'launcher' && 'Click to place launcher'}
              {mode === 'zone' && 'Click & drag to draw zone'}
            </p>
          </div>
        )}
      </div>

      {/* Quick Tips */}
      <div className="mp-tips">
        <span>Del: Delete</span>
        <span>Esc: Deselect</span>
      </div>
    </div>
  );
}
