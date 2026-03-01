/**
 * MissionToolbar — Compact horizontal control bar
 *
 * Scenario, guidance law, nav constant, evasion, entity counts,
 * WTA algorithm, and action buttons (LAUNCH / ABORT / REC / ADV).
 */

import type {
  Scenario,
  GuidanceLaw,
  EvasionType,
  WTAAlgorithm,
} from '../types';

interface MissionToolbarProps {
  connected: boolean;
  isRunning: boolean;
  // Selection state
  selectedScenario: string;
  selectedGuidance: string;
  navConstant: number;
  selectedEvasion: string;
  numInterceptors: number;
  numTargets: number;
  selectedWTA: string;
  // Data
  scenarios: Record<string, Scenario>;
  guidanceLaws: GuidanceLaw[];
  evasionTypes: EvasionType[];
  wtaAlgorithms: WTAAlgorithm[];
  // Setters
  onScenario: (v: string) => void;
  onGuidance: (v: string) => void;
  onNavConstant: (v: number) => void;
  onEvasion: (v: string) => void;
  onNumInterceptors: (v: number) => void;
  onNumTargets: (v: number) => void;
  onWTA: (v: string) => void;
  // Actions
  onLaunch: () => void;
  onAbort: () => void;
  // Recording
  isRecording: boolean;
  onToggleRecording: () => void;
  // Advanced toggle
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
}

export function MissionToolbar({
  connected,
  isRunning,
  selectedScenario,
  selectedGuidance,
  navConstant,
  selectedEvasion,
  numInterceptors,
  numTargets,
  selectedWTA,
  scenarios,
  guidanceLaws,
  evasionTypes,
  wtaAlgorithms,
  onScenario,
  onGuidance,
  onNavConstant,
  onEvasion,
  onNumInterceptors,
  onNumTargets,
  onWTA,
  onLaunch,
  onAbort,
  isRecording,
  onToggleRecording,
  showAdvanced,
  onToggleAdvanced,
}: MissionToolbarProps) {
  return (
    <div className="mission-toolbar">
      <div className="toolbar-group">
        <label>SCENARIO</label>
        <select
          value={selectedScenario}
          onChange={(e) => onScenario(e.target.value)}
          disabled={isRunning}
        >
          {Object.entries(scenarios).map(([name, info]) => (
            <option key={name} value={name} title={info.description}>
              {name.replace('_', ' ').toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <div className="toolbar-group">
        <label>GUIDANCE</label>
        <select
          value={selectedGuidance}
          onChange={(e) => onGuidance(e.target.value)}
          disabled={isRunning}
        >
          {guidanceLaws.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </select>
      </div>

      {selectedGuidance !== 'pure_pursuit' && (
        <div className="toolbar-group nav-group">
          <label>N={navConstant.toFixed(1)}</label>
          <input
            type="range"
            min="1"
            max="8"
            step="0.5"
            value={navConstant}
            onChange={(e) => onNavConstant(parseFloat(e.target.value))}
            disabled={isRunning}
          />
        </div>
      )}

      <div className="toolbar-group">
        <label>EVASION</label>
        <select
          value={selectedEvasion}
          onChange={(e) => onEvasion(e.target.value)}
          disabled={isRunning}
        >
          {evasionTypes.map((e) => (
            <option key={e.id} value={e.id}>
              {e.name}
            </option>
          ))}
        </select>
      </div>

      <div className="toolbar-group interceptor-group">
        <label>INT: {numInterceptors}</label>
        <input
          type="range"
          min="1"
          max="8"
          step="1"
          value={numInterceptors}
          onChange={(e) => onNumInterceptors(parseInt(e.target.value))}
          disabled={isRunning}
        />
      </div>

      <div className="toolbar-group target-group">
        <label>TGT: {numTargets}</label>
        <input
          type="range"
          min="1"
          max="4"
          step="1"
          value={numTargets}
          onChange={(e) => onNumTargets(parseInt(e.target.value))}
          disabled={isRunning}
        />
      </div>

      {numTargets > 1 && (
        <div className="toolbar-group">
          <label>WTA</label>
          <select
            value={selectedWTA}
            onChange={(e) => onWTA(e.target.value)}
            disabled={isRunning}
          >
            {wtaAlgorithms.map((alg) => (
              <option key={alg.id} value={alg.id} title={alg.description}>
                {alg.name}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="toolbar-actions">
        {!isRunning ? (
          <button className="btn-launch" onClick={onLaunch} disabled={!connected}>
            LAUNCH
          </button>
        ) : (
          <button className="btn-abort" onClick={onAbort}>
            ABORT
          </button>
        )}

        <button
          className={`btn-record ${isRecording ? 'recording' : ''}`}
          onClick={onToggleRecording}
          title={isRecording ? 'Stop Recording' : 'Start Recording'}
        >
          REC
        </button>

        <button
          className={`btn-advanced ${showAdvanced ? 'active' : ''}`}
          onClick={onToggleAdvanced}
        >
          ADV
        </button>
      </div>
    </div>
  );
}
