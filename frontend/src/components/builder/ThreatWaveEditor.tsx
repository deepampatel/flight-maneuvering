/**
 * ThreatWaveEditor — List of wave configurations with add/remove
 */

import { WaveCard } from './WaveCard';
import type { BuilderWaveConfig } from '../../types';

interface ThreatWaveEditorProps {
  waves: BuilderWaveConfig[];
  onChange: (waves: BuilderWaveConfig[]) => void;
  disabled: boolean;
}

let waveCounter = 0;

function createDefaultWave(delay = 0): BuilderWaveConfig {
  waveCounter++;
  return {
    id: `wave_${Date.now()}_${waveCounter}`,
    delay,
    threat_type: 'qassam',
    count: 3,
    spawn_bearing: 0,
    spawn_range: 15000,
    spawn_altitude: 2000,
    spacing: 300,
    salvo_interval: 0.5,
    decoy_fraction: 0,
  };
}

export function ThreatWaveEditor({ waves, onChange, disabled }: ThreatWaveEditorProps) {
  const addWave = () => {
    const lastDelay = waves.length > 0 ? waves[waves.length - 1].delay + 10 : 0;
    onChange([...waves, createDefaultWave(lastDelay)]);
  };

  const updateWave = (index: number, updated: BuilderWaveConfig) => {
    const next = [...waves];
    next[index] = updated;
    onChange(next);
  };

  const removeWave = (index: number) => {
    onChange(waves.filter((_, i) => i !== index));
  };

  return (
    <div className="threat-wave-editor">
      {waves.length === 0 && (
        <div className="builder-empty">No waves configured</div>
      )}
      {waves.map((wave, i) => (
        <WaveCard
          key={wave.id}
          wave={wave}
          index={i}
          onChange={(u) => updateWave(i, u)}
          onRemove={() => removeWave(i)}
          disabled={disabled}
        />
      ))}
      <button className="builder-add-btn" onClick={addWave} disabled={disabled}>
        + Add Wave
      </button>
    </div>
  );
}
