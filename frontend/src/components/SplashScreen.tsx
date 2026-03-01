/**
 * SplashScreen — Minimal loading overlay
 *
 * Shows on first load with progress bar and smooth fade-out transition.
 */

import { useState, useEffect } from 'react';

interface SplashScreenProps {
  connected: boolean;
  onComplete: () => void;
}

export function SplashScreen({ connected, onComplete }: SplashScreenProps) {
  const [fadeOut, setFadeOut] = useState(false);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setMinTimeElapsed(true), 1500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (connected && minTimeElapsed) {
      setFadeOut(true);
    }
  }, [connected, minTimeElapsed]);

  useEffect(() => {
    if (fadeOut) {
      const t = setTimeout(onComplete, 400);
      return () => clearTimeout(t);
    }
  }, [fadeOut, onComplete]);

  return (
    <div className={`splash-screen ${fadeOut ? 'splash-fade-out' : ''}`}>
      <div className="splash-content">
        <h1 className="splash-logo">INTERCEPT</h1>
        <div className="splash-progress-track">
          <div className="splash-progress-bar" />
        </div>
        <div className="splash-status-text">
          {connected ? 'Connected' : 'Connecting...'}
        </div>
      </div>
    </div>
  );
}
