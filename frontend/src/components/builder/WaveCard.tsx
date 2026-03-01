/**
 * WaveCard — Configuration card for a single threat wave
 */

import { THREAT_TYPES } from '../../data/tierDefaults';
import type { BuilderWaveConfig } from '../../types';

interface WaveCardProps {
  wave: BuilderWaveConfig;
  index: number;
  onChange: (updated: BuilderWaveConfig) => void;
  onRemove: () => void;
  disabled: boolean;
}

export function WaveCard({ wave, index, onChange, onRemove, disabled }: WaveCardProps) {
  const set = <K extends keyof BuilderWaveConfig>(key: K, value: BuilderWaveConfig[K]) => {
    onChange({ ...wave, [key]: value });
  };

  const threatInfo = THREAT_TYPES[wave.threat_type];

  return (
    <div className="wave-card">
      <div className="wave-card-header">
        <span className="wave-label">Wave {index + 1}</span>
        <span className="wave-summary">
          {wave.count}× {threatInfo?.name || wave.threat_type} @ T+{wave.delay}s
        </span>
        <button className="wave-remove" onClick={onRemove} disabled={disabled}>&times;</button>
      </div>

      <div className="wave-fields">
        <div className="wave-field">
          <label>Threat</label>
          <select value={wave.threat_type} onChange={(e) => set('threat_type', e.target.value)} disabled={disabled}>
            {Object.entries(THREAT_TYPES).map(([id, info]) => (
              <option key={id} value={id}>{info.name} ({info.speed}m/s)</option>
            ))}
          </select>
        </div>

        <div className="wave-field">
          <label>Count</label>
          <input type="number" min={1} max={20} value={wave.count}
            onChange={(e) => set('count', +e.target.value)} disabled={disabled} />
        </div>

        <div className="wave-field">
          <label>Delay {wave.delay}s</label>
          <input type="range" min={0} max={120} step={1} value={wave.delay}
            onChange={(e) => set('delay', +e.target.value)} disabled={disabled} />
        </div>

        <div className="wave-field">
          <label>Bearing {wave.spawn_bearing}°</label>
          <input type="range" min={0} max={360} step={5} value={wave.spawn_bearing}
            onChange={(e) => set('spawn_bearing', +e.target.value)} disabled={disabled} />
        </div>

        <div className="wave-field">
          <label>Range {(wave.spawn_range / 1000).toFixed(0)}km</label>
          <input type="range" min={5000} max={100000} step={1000} value={wave.spawn_range}
            onChange={(e) => set('spawn_range', +e.target.value)} disabled={disabled} />
        </div>

        <div className="wave-field">
          <label>Decoys {(wave.decoy_fraction * 100).toFixed(0)}%</label>
          <input type="range" min={0} max={0.5} step={0.05} value={wave.decoy_fraction}
            onChange={(e) => set('decoy_fraction', +e.target.value)} disabled={disabled} />
        </div>
      </div>
    </div>
  );
}
