/**
 * WelcomeModal — First-visit onboarding walkthrough
 *
 * Three-step briefing-style tutorial:
 *   1. SELECT A MISSION
 *   2. CONFIGURE & LAUNCH
 *   3. OBSERVE & ANALYZE
 *
 * Uses localStorage to track whether the user has been onboarded.
 * Military briefing aesthetic matching ScenarioBriefing.
 */

import { useState, useEffect } from 'react';

const ONBOARDED_KEY = 'intercept_onboarded';

interface WelcomeModalProps {
  forceShow?: boolean;
}

const STEPS = [
  {
    title: 'SELECT A MISSION',
    icon: '\u25CE', // ◎
    description:
      'Choose a scenario from the top bar buttons. Each scenario has a difficulty level and unique tactical challenge. Click any scenario to see a full mission briefing before launch.',
    detail: 'GREEN = Easy \u00b7 YELLOW = Medium \u00b7 ORANGE = Hard \u00b7 RED = Extreme',
  },
  {
    title: 'CONFIGURE & LAUNCH',
    icon: '\u25B2', // ▲
    description:
      'Use the toolbar to select guidance law, adjust navigation constant, set evasion type, and configure interceptor/target counts. Press LAUNCH when ready.',
    detail: 'ADV opens advanced panels: Monte Carlo analysis, environment, swarm tactics, and more.',
  },
  {
    title: 'OBSERVE & ANALYZE',
    icon: '\u229A', // ⊚
    description:
      'Watch the 3D engagement unfold in real-time. Click entities to select them and view telemetry. Use camera modes (FREE / CHASE / TAC / CIN) for different perspectives.',
    detail: 'Press REC to record. Replay past engagements in the Replay Theater.',
  },
];

export function WelcomeModal({ forceShow }: WelcomeModalProps) {
  const [step, setStep] = useState(0);
  const [visible, setVisible] = useState(false);
  const [dontShow, setDontShow] = useState(false);

  useEffect(() => {
    if (forceShow) {
      setVisible(true);
      return;
    }
    const seen = localStorage.getItem(ONBOARDED_KEY);
    if (!seen) setVisible(true);
  }, [forceShow]);

  const handleClose = () => {
    setVisible(false);
    if (dontShow || step === STEPS.length - 1) {
      localStorage.setItem(ONBOARDED_KEY, '1');
    }
  };

  const handleNext = () => {
    if (step < STEPS.length - 1) {
      setStep(step + 1);
    } else {
      handleClose();
    }
  };

  if (!visible) return null;

  const current = STEPS[step];

  return (
    <div className="welcome-backdrop" onClick={handleClose}>
      <div className="welcome-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="welcome-header">
          <div className="welcome-badge">MISSION BRIEFING</div>
          <div className="welcome-step-indicator">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className={`welcome-step-dot ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}
              />
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="welcome-body">
          <div className="welcome-icon">{current.icon}</div>
          <h2 className="welcome-title">{current.title}</h2>
          <p className="welcome-desc">{current.description}</p>
          <div className="welcome-detail">{current.detail}</div>
        </div>

        {/* Footer */}
        <div className="welcome-footer">
          <label className="welcome-dont-show">
            <input
              type="checkbox"
              checked={dontShow}
              onChange={(e) => setDontShow(e.target.checked)}
            />
            Don&apos;t show again
          </label>

          <div className="welcome-actions">
            <button className="welcome-btn-skip" onClick={handleClose}>
              SKIP
            </button>
            <button className="welcome-btn-next" onClick={handleNext}>
              {step < STEPS.length - 1 ? 'NEXT' : 'BEGIN'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
