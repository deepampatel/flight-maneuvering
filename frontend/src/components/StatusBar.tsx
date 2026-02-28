/**
 * StatusBar - Thin system metrics bar below the header
 *
 * Shows UTC clock, sim tick, entity count, recording status, and FPS.
 * Military C2 aesthetic — amber/green monospace readouts.
 */

import { useState, useEffect, useRef } from 'react';
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
  const [fps, setFps] = useState(0);
  const frameTimesRef = useRef<number[]>([]);
  const lastTickRef = useRef<number>(0);
  const lastTickTimeRef = useRef<number>(0);
  const [tickRate, setTickRate] = useState(0);

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

  // FPS counter
  useEffect(() => {
    let animId: number;
    const measure = () => {
      const now = performance.now();
      frameTimesRef.current.push(now);
      // Keep last 60 frame times
      if (frameTimesRef.current.length > 60) {
        frameTimesRef.current.shift();
      }
      if (frameTimesRef.current.length > 1) {
        const elapsed = now - frameTimesRef.current[0];
        setFps(Math.round((frameTimesRef.current.length - 1) / (elapsed / 1000)));
      }
      animId = requestAnimationFrame(measure);
    };
    animId = requestAnimationFrame(measure);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Tick rate calculation
  useEffect(() => {
    if (state?.tick && state.tick !== lastTickRef.current) {
      const now = performance.now();
      if (lastTickTimeRef.current > 0) {
        const dt = (now - lastTickTimeRef.current) / 1000;
        const dTick = state.tick - lastTickRef.current;
        if (dt > 0 && dt < 2) {
          setTickRate(Math.round(dTick / dt));
        }
      }
      lastTickRef.current = state.tick;
      lastTickTimeRef.current = now;
    }
  }, [state?.tick]);

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
          <span className="sb-label">SIM TIME</span>
          <span className="sb-value">{simTime}s</span>
        </span>
      </div>

      <div className="sb-section">
        <span className="sb-item">
          <span className="sb-label">TICK</span>
          <span className="sb-value">{state?.tick || 0}</span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">TICK/s</span>
          <span className="sb-value">{tickRate}</span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">ENTITIES</span>
          <span className="sb-value sb-value-accent">{entityCount}</span>
        </span>
        <span className="sb-divider" />
        <span className="sb-item">
          <span className="sb-label">FPS</span>
          <span className={`sb-value ${fps < 30 ? 'sb-value-warn' : ''}`}>{fps}</span>
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
          <button className="sb-mute-btn" onClick={onToggleMute} title={muted ? 'Unmute' : 'Mute'}>
            {muted ? '🔇' : '🔊'}
          </button>
        )}
      </div>
    </div>
  );
}
