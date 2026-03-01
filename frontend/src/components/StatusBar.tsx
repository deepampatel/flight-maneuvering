/**
 * StatusBar - Thin system metrics bar
 *
 * Shows connection status, sim time, entity count, and recording indicator.
 * Dev metrics (FPS, tick rate) hidden by default.
 */

import { useState, useEffect } from 'react';
import type { SimStateEvent } from '../types';

interface StatusBarProps {
  state: SimStateEvent | null;
  connected: boolean;
  isRecording: boolean;
  muted?: boolean;
  onToggleMute?: () => void;
}

export function StatusBar({ state, connected, isRecording, muted, onToggleMute }: StatusBarProps) {
  const [clock, setClock] = useState('');

  // UTC clock - update every second
  useEffect(() => {
    const update = () => {
      const now = new Date();
      setClock(now.toISOString().slice(11, 19));
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  const entityCount = state?.entities?.length || 0;
  const simTime = state?.sim_time ? state.sim_time.toFixed(1) : '0.0';
  const status = state?.status || 'IDLE';

  return (
    <div className="status-bar">
      <div className="sb-section">
        <span className="sb-item">
          <span className="sb-label">UTC</span>
          <span className="sb-value">{clock}</span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">STATUS</span>
          <span className={`sb-value sb-status-${status.toLowerCase()}`}>
            {status.toUpperCase()}
          </span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">T+</span>
          <span className="sb-value">{simTime}s</span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">ENTITIES</span>
          <span className="sb-value">{entityCount}</span>
        </span>
      </div>

      <div className="sb-section sb-section-right">
        {isRecording && (
          <span className="sb-item sb-recording">
            <span className="sb-rec-dot" />
            REC
          </span>
        )}
        <span className={`sb-item sb-conn ${connected ? 'sb-conn-on' : 'sb-conn-off'}`}>
          {connected ? 'LINK' : 'NO LINK'}
        </span>
        {onToggleMute && (
          <button
            className="sb-mute-btn"
            onClick={onToggleMute}
            title={muted ? 'Unmute audio' : 'Mute audio'}
          >
            {muted ? 'OFF' : 'ON'}
          </button>
        )}
      </div>
    </div>
  );
}
