/**
 * ScenarioPresets — Card grid of narrative scenarios + Custom option
 */

import { NARRATIVE_SCENARIOS, type NarrativeScenario } from '../../data/scenarios';

interface ScenarioPresetsProps {
  activeId: string | null;
  onSelect: (scenario: NarrativeScenario | null) => void;
  disabled: boolean;
}

const DIFFICULTY_COLORS: Record<string, string> = {
  EASY: '#22c55e',
  MEDIUM: '#eab308',
  HARD: '#ef4444',
  EXTREME: '#8b5cf6',
};

export function ScenarioPresets({ activeId, onSelect, disabled }: ScenarioPresetsProps) {
  return (
    <div className="builder-presets">
      <button
        className={`preset-card preset-custom ${activeId === null ? 'active' : ''}`}
        onClick={() => onSelect(null)}
        disabled={disabled}
      >
        <span className="preset-icon">+</span>
        <span className="preset-name">Custom</span>
      </button>
      {NARRATIVE_SCENARIOS.map((s) => (
        <button
          key={s.id}
          className={`preset-card ${activeId === s.id ? 'active' : ''}`}
          onClick={() => onSelect(s)}
          disabled={disabled}
          title={s.briefing.situation}
        >
          <span
            className="preset-difficulty"
            style={{ color: DIFFICULTY_COLORS[s.difficulty] }}
          >
            {s.difficulty}
          </span>
          <span className="preset-name">{s.name}</span>
          <span className="preset-codename">{s.codename}</span>
        </button>
      ))}
    </div>
  );
}
