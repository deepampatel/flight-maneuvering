/**
 * EngagementSettings — Guidance, WTA, kill radius, entity counts
 */

interface EngagementSettingsProps {
  guidance: string;
  navConstant: number;
  evasion: string;
  wtaAlgorithm: string;
  killRadius: number;
  numInterceptors: number;
  numTargets: number;
  onChange: (field: string, value: string | number) => void;
  disabled: boolean;
}

export function EngagementSettings({
  guidance, navConstant, evasion, wtaAlgorithm, killRadius,
  numInterceptors, numTargets, onChange, disabled,
}: EngagementSettingsProps) {
  return (
    <div className="engagement-settings">
      <div className="setting-row">
        <label>Guidance</label>
        <select value={guidance} onChange={(e) => onChange('guidance', e.target.value)} disabled={disabled}>
          <option value="pure_pursuit">Pure Pursuit</option>
          <option value="proportional_nav">Proportional Nav</option>
          <option value="augmented_pn">Augmented PN</option>
        </select>
      </div>

      {guidance !== 'pure_pursuit' && (
        <div className="setting-row">
          <label>Nav Constant N={navConstant.toFixed(1)}</label>
          <input type="range" min={1} max={8} step={0.5} value={navConstant}
            onChange={(e) => onChange('navConstant', +e.target.value)} disabled={disabled} />
        </div>
      )}

      <div className="setting-row">
        <label>Evasion</label>
        <select value={evasion} onChange={(e) => onChange('evasion', e.target.value)} disabled={disabled}>
          <option value="none">None</option>
          <option value="constant_turn">Constant Turn</option>
          <option value="weave">Weave</option>
          <option value="barrel_roll">Barrel Roll</option>
          <option value="random_jink">Random Jink</option>
        </select>
      </div>

      <div className="setting-row">
        <label>WTA Algorithm</label>
        <select value={wtaAlgorithm} onChange={(e) => onChange('wtaAlgorithm', e.target.value)} disabled={disabled}>
          <option value="hungarian">Hungarian (Optimal)</option>
          <option value="greedy_nearest">Greedy Nearest</option>
          <option value="greedy_threat">Greedy Threat</option>
          <option value="round_robin">Round Robin</option>
        </select>
      </div>

      <div className="setting-row">
        <label>Kill radius {killRadius}m</label>
        <input type="range" min={10} max={500} step={10} value={killRadius}
          onChange={(e) => onChange('killRadius', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-row">
        <label>Interceptors {numInterceptors}</label>
        <input type="range" min={0} max={8} step={1} value={numInterceptors}
          onChange={(e) => onChange('numInterceptors', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-row">
        <label>Targets {numTargets}</label>
        <input type="range" min={1} max={8} step={1} value={numTargets}
          onChange={(e) => onChange('numTargets', +e.target.value)} disabled={disabled} />
      </div>
    </div>
  );
}
