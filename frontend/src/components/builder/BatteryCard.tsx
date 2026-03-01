/**
 * BatteryCard — Configuration card for a single defense battery
 */

import { useState } from 'react';
import { TIER_DEFAULTS, TIER_IDS, type TierId } from '../../data/tierDefaults';
import type { BuilderBatteryConfig } from '../../types';

interface BatteryCardProps {
  battery: BuilderBatteryConfig;
  onChange: (updated: BuilderBatteryConfig) => void;
  onRemove: () => void;
  disabled: boolean;
}

export function BatteryCard({ battery, onChange, onRemove, disabled }: BatteryCardProps) {
  const [expanded, setExpanded] = useState(false);
  const tier = TIER_DEFAULTS[battery.tier];

  const setTier = (tierId: string) => {
    const t = TIER_DEFAULTS[tierId];
    if (!t) return;
    onChange({
      ...battery,
      tier: tierId,
      radar_range: t.defaults.radar_range,
      radar_sector: t.defaults.radar_sector,
      num_launchers: t.defaults.num_launchers,
      missiles_per_launcher: t.defaults.missiles_per_launcher,
      max_simultaneous: t.defaults.max_simultaneous,
      min_range: t.minRange,
      max_range: t.maxRange,
      launch_speed: t.defaults.launch_speed,
      launch_elevation: t.defaults.launch_elevation,
      min_altitude: t.defaults.min_altitude,
    });
  };

  const set = <K extends keyof BuilderBatteryConfig>(key: K, value: BuilderBatteryConfig[K]) => {
    onChange({ ...battery, [key]: value });
  };

  const totalMissiles = battery.num_launchers * battery.missiles_per_launcher;

  return (
    <div className="battery-card" style={{ borderLeftColor: tier?.color || '#3b82f6' }}>
      <div className="battery-card-header">
        <input
          className="battery-name-input"
          value={battery.name}
          onChange={(e) => set('name', e.target.value)}
          disabled={disabled}
        />
        <button className="battery-remove" onClick={onRemove} disabled={disabled}>&times;</button>
      </div>

      {/* Tier selector */}
      <div className="battery-tier-row">
        {TIER_IDS.map((tid) => {
          const t = TIER_DEFAULTS[tid as TierId];
          return (
            <button
              key={tid}
              className={`tier-btn ${battery.tier === tid ? 'active' : ''}`}
              style={{
                '--tier-color': t.color,
                borderColor: battery.tier === tid ? t.color : undefined,
                color: battery.tier === tid ? t.color : undefined,
              } as React.CSSProperties}
              onClick={() => setTier(tid)}
              disabled={disabled}
            >
              {t.name}
            </button>
          );
        })}
      </div>

      {/* Position */}
      <div className="battery-field-row">
        <label>Position</label>
        <div className="battery-pos-inputs">
          <input
            type="number" placeholder="X"
            value={battery.position.x} disabled={disabled}
            onChange={(e) => set('position', { ...battery.position, x: +e.target.value })}
          />
          <input
            type="number" placeholder="Z"
            value={battery.position.y} disabled={disabled}
            onChange={(e) => set('position', { ...battery.position, y: +e.target.value })}
          />
        </div>
      </div>

      {/* Radar range */}
      <div className="battery-field-row">
        <label>Radar {(battery.radar_range / 1000).toFixed(0)}km</label>
        <input
          type="range"
          min={tier?.minRange || 4000}
          max={(tier?.maxRange || 70000) * 1.5}
          step={1000}
          value={battery.radar_range}
          onChange={(e) => set('radar_range', +e.target.value)}
          disabled={disabled}
        />
      </div>

      {/* Ammo */}
      <div className="battery-field-row">
        <label>Ammo {totalMissiles}</label>
        <div className="battery-ammo-inputs">
          <span className="ammo-detail">{battery.num_launchers} launchers</span>
          <span className="ammo-detail">&times; {battery.missiles_per_launcher} ea</span>
        </div>
      </div>
      <div className="battery-field-row">
        <label>Launchers</label>
        <input type="range" min={1} max={8} step={1}
          value={battery.num_launchers} disabled={disabled}
          onChange={(e) => set('num_launchers', +e.target.value)} />
      </div>
      <div className="battery-field-row">
        <label>Per launcher</label>
        <input type="range" min={2} max={40} step={2}
          value={battery.missiles_per_launcher} disabled={disabled}
          onChange={(e) => set('missiles_per_launcher', +e.target.value)} />
      </div>

      {/* Advanced toggle */}
      <button className="battery-advanced-toggle" onClick={() => setExpanded(!expanded)}>
        {expanded ? '▾ Less' : '▸ Advanced'}
      </button>

      {expanded && (
        <div className="battery-advanced">
          <div className="battery-field-row">
            <label>Sector {battery.radar_sector}°</label>
            <input type="range" min={30} max={360} step={10}
              value={battery.radar_sector} disabled={disabled}
              onChange={(e) => set('radar_sector', +e.target.value)} />
          </div>
          <div className="battery-field-row">
            <label>Max simul.</label>
            <input type="number" min={1} max={20}
              value={battery.max_simultaneous} disabled={disabled}
              onChange={(e) => set('max_simultaneous', +e.target.value)} />
          </div>
          <div className="battery-field-row">
            <label>Launch spd</label>
            <input type="number" min={100} max={3000} step={50}
              value={battery.launch_speed} disabled={disabled}
              onChange={(e) => set('launch_speed', +e.target.value)} />
          </div>
          <div className="battery-field-row">
            <label>Min alt</label>
            <input type="number" min={0} max={50000} step={100}
              value={battery.min_altitude} disabled={disabled}
              onChange={(e) => set('min_altitude', +e.target.value)} />
          </div>
        </div>
      )}
    </div>
  );
}
