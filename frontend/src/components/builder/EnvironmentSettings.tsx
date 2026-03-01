/**
 * EnvironmentSettings — Wind, drag, terrain, datalink, HMT, max time
 */

interface EnvironmentSettingsProps {
  windSpeed: number;
  windDirection: number;
  windGusts: number;
  enableDrag: boolean;
  enableTerrain: boolean;
  enableDatalink: boolean;
  enableCooperative: boolean;
  enableSwarm: boolean;
  enableHmt: boolean;
  hmtAuthorityLevel: string;
  maxTime: number;
  onChange: (field: string, value: string | number | boolean) => void;
  disabled: boolean;
}

export function EnvironmentSettings({
  windSpeed, windDirection, windGusts, enableDrag,
  enableTerrain, enableDatalink, enableCooperative,
  enableSwarm, enableHmt, hmtAuthorityLevel, maxTime,
  onChange, disabled,
}: EnvironmentSettingsProps) {
  return (
    <div className="environment-settings">
      <div className="setting-row">
        <label>Wind {windSpeed.toFixed(0)} m/s</label>
        <input type="range" min={0} max={30} step={1} value={windSpeed}
          onChange={(e) => onChange('windSpeed', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-row">
        <label>Direction {windDirection}°</label>
        <input type="range" min={0} max={360} step={5} value={windDirection}
          onChange={(e) => onChange('windDirection', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-row">
        <label>Gusts {windGusts.toFixed(0)} m/s</label>
        <input type="range" min={0} max={15} step={1} value={windGusts}
          onChange={(e) => onChange('windGusts', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-row">
        <label>Max time {maxTime}s</label>
        <input type="range" min={15} max={300} step={5} value={maxTime}
          onChange={(e) => onChange('maxTime', +e.target.value)} disabled={disabled} />
      </div>

      <div className="setting-toggles">
        <div className="toggle-row">
          <label>Aerodynamic drag</label>
          <input type="checkbox" className="toggle-switch" checked={enableDrag} disabled={disabled}
            onChange={(e) => onChange('enableDrag', e.target.checked)} />
        </div>
        <div className="toggle-row">
          <label>Terrain masking</label>
          <input type="checkbox" className="toggle-switch" checked={enableTerrain} disabled={disabled}
            onChange={(e) => onChange('enableTerrain', e.target.checked)} />
        </div>
        <div className="toggle-row">
          <label>Datalink simulation</label>
          <input type="checkbox" className="toggle-switch" checked={enableDatalink} disabled={disabled}
            onChange={(e) => onChange('enableDatalink', e.target.checked)} />
        </div>
        <div className="toggle-row">
          <label>Cooperative engagement</label>
          <input type="checkbox" className="toggle-switch" checked={enableCooperative} disabled={disabled}
            onChange={(e) => onChange('enableCooperative', e.target.checked)} />
        </div>
        <div className="toggle-row">
          <label>Swarm formation</label>
          <input type="checkbox" className="toggle-switch" checked={enableSwarm} disabled={disabled}
            onChange={(e) => onChange('enableSwarm', e.target.checked)} />
        </div>
        <div className="toggle-row">
          <label>Human-Machine Teaming</label>
          <input type="checkbox" className="toggle-switch" checked={enableHmt} disabled={disabled}
            onChange={(e) => onChange('enableHmt', e.target.checked)} />
        </div>
      </div>

      {enableHmt && (
        <div className="setting-row">
          <label>Authority</label>
          <select value={hmtAuthorityLevel} disabled={disabled}
            onChange={(e) => onChange('hmtAuthorityLevel', e.target.value)}>
            <option value="full_auto">Full Auto</option>
            <option value="human_on_loop">Human on Loop</option>
            <option value="human_in_loop">Human in Loop</option>
            <option value="manual">Manual</option>
          </select>
        </div>
      )}
    </div>
  );
}
