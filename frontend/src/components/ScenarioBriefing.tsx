/**
 * ScenarioBriefing - Full-screen mission briefing modal
 *
 * Military C2 aesthetic with typewriter text animation,
 * threat diagram, and classification header.
 */

import type { NarrativeScenario } from '../data/scenarios';

interface ScenarioBriefingProps {
  scenario: NarrativeScenario;
  onCommence: () => void;
  onBack: () => void;
}

const DIFFICULTY_COLORS: Record<string, string> = {
  EASY: '#4ade80',
  MEDIUM: '#fbbf24',
  HARD: '#f97316',
  EXTREME: '#ef4444',
};

export function ScenarioBriefing({ scenario, onCommence, onBack }: ScenarioBriefingProps) {
  const diffColor = DIFFICULTY_COLORS[scenario.difficulty] || '#64748b';

  return (
    <div className="briefing-overlay">
      <div className="briefing-modal">
        {/* Classification header */}
        <div className="briefing-classification">
          TOP SECRET // NOFORN // MISSION BRIEFING
        </div>

        {/* Mission header */}
        <div className="briefing-header">
          <div className="briefing-codename">{scenario.codename}</div>
          <h2 className="briefing-title">{scenario.name}</h2>
          <span className="briefing-difficulty" style={{ color: diffColor, borderColor: diffColor }}>
            {scenario.difficulty}
          </span>
        </div>

        {/* Situation */}
        <div className="briefing-section">
          <div className="briefing-section-label">SITUATION</div>
          <p className="briefing-text typewriter">{scenario.briefing.situation}</p>
        </div>

        {/* Objective */}
        <div className="briefing-section">
          <div className="briefing-section-label">OBJECTIVE</div>
          <p className="briefing-text">{scenario.briefing.objective}</p>
        </div>

        {/* Threat Picture */}
        <div className="briefing-section">
          <div className="briefing-section-label">THREAT ASSESSMENT</div>
          <p className="briefing-text briefing-threat">{scenario.briefing.threatPicture}</p>
        </div>

        {/* Constraints */}
        <div className="briefing-section">
          <div className="briefing-section-label">CONSTRAINTS</div>
          <ul className="briefing-constraints">
            {scenario.briefing.constraints.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>

        {/* Config summary */}
        <div className="briefing-config">
          <span>Interceptors: {scenario.config.numInterceptors}</span>
          <span>Targets: {scenario.config.numTargets}</span>
          <span>Guidance: {scenario.config.guidance.replace(/_/g, ' ')}</span>
          <span>Par Time: {scenario.scoring.parTime}s</span>
        </div>

        {/* Actions */}
        <div className="briefing-actions">
          <button className="briefing-btn-back" onClick={onBack}>ABORT</button>
          <button className="briefing-btn-commence" onClick={onCommence}>COMMENCE</button>
        </div>

        {/* Classification footer */}
        <div className="briefing-classification">
          TOP SECRET // NOFORN
        </div>
      </div>
    </div>
  );
}

/**
 * ScenarioDebrief - Post-engagement results modal
 */
interface ScenarioDebriefProps {
  scenario: NarrativeScenario;
  result: {
    total: number;
    grade: string;
    breakdown: { time: number; accuracy: number; efficiency: number };
  };
  simTime: number;
  missDistance: number;
  onReplay: () => void;
  onNext: () => void;
  onBack: () => void;
}

const GRADE_COLORS: Record<string, string> = {
  S: '#a855f7',
  A: '#4ade80',
  B: '#60a5fa',
  C: '#fbbf24',
  D: '#f97316',
  F: '#ef4444',
};

export function ScenarioDebrief({ scenario, result, simTime, missDistance, onReplay, onNext, onBack }: ScenarioDebriefProps) {
  const gradeColor = GRADE_COLORS[result.grade] || '#64748b';

  return (
    <div className="briefing-overlay">
      <div className="briefing-modal debrief-modal">
        <div className="briefing-classification">
          ENGAGEMENT REPORT // {scenario.codename}
        </div>

        {/* Grade */}
        <div className="debrief-grade" style={{ color: gradeColor, textShadow: `0 0 30px ${gradeColor}` }}>
          {result.grade}
        </div>
        <div className="debrief-score">{result.total} POINTS</div>

        {/* Stats */}
        <div className="debrief-stats">
          <div className="debrief-stat">
            <span className="debrief-stat-label">TIME</span>
            <span className="debrief-stat-value">{simTime.toFixed(1)}s</span>
            <span className="debrief-stat-sub">par: {scenario.scoring.parTime}s</span>
          </div>
          <div className="debrief-stat">
            <span className="debrief-stat-label">MISS DIST</span>
            <span className="debrief-stat-value">{missDistance.toFixed(0)}m</span>
          </div>
        </div>

        {/* Score breakdown */}
        <div className="debrief-breakdown">
          <div className="debrief-bar-group">
            <span className="debrief-bar-label">TIME BONUS</span>
            <div className="debrief-bar">
              <div className="debrief-bar-fill" style={{ width: `${Math.min(result.breakdown.time, 100)}%`, background: '#4ade80' }} />
            </div>
            <span className="debrief-bar-value">+{result.breakdown.time}</span>
          </div>
          <div className="debrief-bar-group">
            <span className="debrief-bar-label">ACCURACY</span>
            <div className="debrief-bar">
              <div className="debrief-bar-fill" style={{ width: `${Math.min(result.breakdown.accuracy, 100)}%`, background: '#60a5fa' }} />
            </div>
            <span className="debrief-bar-value">+{result.breakdown.accuracy}</span>
          </div>
          <div className="debrief-bar-group">
            <span className="debrief-bar-label">EFFICIENCY</span>
            <div className="debrief-bar">
              <div className="debrief-bar-fill" style={{ width: `${Math.min(result.breakdown.efficiency, 100)}%`, background: '#fbbf24' }} />
            </div>
            <span className="debrief-bar-value">+{result.breakdown.efficiency}</span>
          </div>
        </div>

        <div className="briefing-actions">
          <button className="briefing-btn-back" onClick={onBack}>MENU</button>
          <button className="briefing-btn-back" onClick={onReplay}>REPLAY</button>
          <button className="briefing-btn-commence" onClick={onNext}>NEXT MISSION</button>
        </div>
      </div>
    </div>
  );
}
