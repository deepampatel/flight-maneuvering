/**
 * DefenseLayerEditor — List of battery configurations with add/remove
 */

import { BatteryCard } from './BatteryCard';
import { TIER_DEFAULTS } from '../../data/tierDefaults';
import type { BuilderBatteryConfig } from '../../types';

interface DefenseLayerEditorProps {
  batteries: BuilderBatteryConfig[];
  onChange: (batteries: BuilderBatteryConfig[]) => void;
  disabled: boolean;
}

let batteryCounter = 0;

export function createDefaultBattery(tier = 'iron_dome'): BuilderBatteryConfig {
  const t = TIER_DEFAULTS[tier];
  batteryCounter++;
  return {
    id: `bat_${Date.now()}_${batteryCounter}`,
    name: `${t.name} ${String.fromCharCode(64 + batteryCounter)}`,
    tier,
    position: { x: -500 + (batteryCounter - 1) * 400, y: 500, z: 0 },
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
    protected_area_ids: [],
  };
}

export function DefenseLayerEditor({ batteries, onChange, disabled }: DefenseLayerEditorProps) {
  const addBattery = () => {
    onChange([...batteries, createDefaultBattery()]);
  };

  const updateBattery = (index: number, updated: BuilderBatteryConfig) => {
    const next = [...batteries];
    next[index] = updated;
    onChange(next);
  };

  const removeBattery = (index: number) => {
    onChange(batteries.filter((_, i) => i !== index));
  };

  return (
    <div className="defense-layer-editor">
      {batteries.length === 0 && (
        <div className="builder-empty">No batteries configured</div>
      )}
      {batteries.map((bat, i) => (
        <BatteryCard
          key={bat.id}
          battery={bat}
          onChange={(u) => updateBattery(i, u)}
          onRemove={() => removeBattery(i)}
          disabled={disabled}
        />
      ))}
      <button className="builder-add-btn" onClick={addBattery} disabled={disabled}>
        + Add Battery
      </button>
    </div>
  );
}
