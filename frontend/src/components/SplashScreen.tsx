/**
 * SplashScreen — Cinematic loading overlay
 *
 * Shows on first load with animated radar sweep, system initialization
 * status messages, and smooth fade-out transition.
 */

import { useState, useEffect } from 'react';

interface SplashScreenProps {
  connected: boolean;
  onComplete: () => void;
}

const STATUS_MESSAGES = [
  'INITIALIZING SYSTEMS',
  'RADAR SUBSYSTEM ONLINE',
  'SENSOR FUSION ACTIVE',
  'C2 LINK ESTABLISHED',
  'ALL SYSTEMS NOMINAL',
];

export function SplashScreen({ connected, onComplete }: SplashScreenProps) {
  const [statusIndex, setStatusIndex] = useState(0);
  const [fadeOut, setFadeOut] = useState(false);

  // Cycle through status messages
  useEffect(() => {
    const interval = setInterval(() => {
      setStatusIndex((prev) => {
        if (prev < STATUS_MESSAGES.length - 1) return prev + 1;
        return prev;
      });
    }, 400);
    return () => clearInterval(interval);
  }, []);

  // Auto-dismiss: wait for connection + minimum display time
  useEffect(() => {
    const minTime = setTimeout(() => {
      if (connected) {
        setFadeOut(true);
      }
    }, 2000);

    return () => clearTimeout(minTime);
  }, [connected]);

  // If connected arrives after min time
  useEffect(() => {
    if (connected && statusIndex >= STATUS_MESSAGES.length - 1) {
      const t = setTimeout(() => setFadeOut(true), 300);
      return () => clearTimeout(t);
    }
  }, [connected, statusIndex]);

  // Remove from DOM after fade
  useEffect(() => {
    if (fadeOut) {
      const t = setTimeout(onComplete, 600);
      return () => clearTimeout(t);
    }
  }, [fadeOut, onComplete]);

  return (
    <div className={`splash-screen ${fadeOut ? 'splash-fade-out' : ''}`}>
      {/* Radar sweep background */}
      <div className="splash-radar">
        <div className="splash-radar-ring splash-radar-ring-1" />
        <div className="splash-radar-ring splash-radar-ring-2" />
        <div className="splash-radar-ring splash-radar-ring-3" />
        <div className="splash-radar-sweep" />
        <div className="splash-radar-center" />
      </div>

      {/* Logo */}
      <div className="splash-content">
        <div className="splash-logo-group">
          <h1 className="splash-logo">INTERCEPT</h1>
          <div className="splash-subtitle">MULTI-LAYER AIR DEFENSE SIMULATION</div>
        </div>

        {/* Status messages */}
        <div className="splash-status">
          {STATUS_MESSAGES.map((msg, i) => (
            <div
              key={msg}
              className={`splash-status-line ${i <= statusIndex ? 'splash-status-visible' : ''} ${i === statusIndex ? 'splash-status-active' : ''}`}
            >
              <span className="splash-status-dot">{i < statusIndex ? '\u2713' : i === statusIndex ? '\u25CF' : '\u25CB'}</span>
              <span>{msg}</span>
            </div>
          ))}
        </div>

        <div className="splash-version">v2.0</div>
      </div>
    </div>
  );
}
