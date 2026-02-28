/**
 * useAudio - Procedural audio system using Web Audio API
 *
 * Generates all sounds procedurally (no external files needed).
 * Sounds: launch, explosion, radar lock tone, UI clicks, ambient hum.
 */

import { useRef, useState, useCallback } from 'react';

export function useAudio() {
  const ctxRef = useRef<AudioContext | null>(null);
  const ambientRef = useRef<OscillatorNode | null>(null);
  const ambientGainRef = useRef<GainNode | null>(null);
  const [muted, setMuted] = useState(false);

  const getContext = useCallback(() => {
    if (!ctxRef.current || ctxRef.current.state === 'closed') {
      ctxRef.current = new AudioContext();
    }
    if (ctxRef.current.state === 'suspended') {
      ctxRef.current.resume();
    }
    return ctxRef.current;
  }, []);

  const playTone = useCallback((
    frequency: number,
    duration: number,
    type: OscillatorType = 'sine',
    volume: number = 0.08
  ) => {
    if (muted) return;
    try {
      const ctx = getContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = type;
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch {
      // Audio context may not be available
    }
  }, [muted, getContext]);

  const playNoise = useCallback((duration: number, volume: number = 0.08) => {
    if (muted) return;
    try {
      const ctx = getContext();
      const bufferSize = ctx.sampleRate * duration;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);

      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.3));
      }

      const source = ctx.createBufferSource();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      source.buffer = buffer;
      filter.type = 'lowpass';
      filter.frequency.value = 2000;
      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

      source.connect(filter).connect(gain).connect(ctx.destination);
      source.start();
    } catch {
      // Audio context may not be available
    }
  }, [muted, getContext]);

  const playLaunch = useCallback(() => {
    // Rising swoosh + low rumble
    playTone(150, 0.6, 'sawtooth', 0.06);
    setTimeout(() => playTone(300, 0.3, 'sawtooth', 0.04), 100);
    setTimeout(() => playNoise(0.4, 0.05), 50);
  }, [playTone, playNoise]);

  const playExplosion = useCallback(() => {
    // Impact noise + low boom
    playNoise(0.5, 0.12);
    playTone(60, 0.4, 'sine', 0.1);
    setTimeout(() => playTone(40, 0.3, 'sine', 0.06), 100);
  }, [playTone, playNoise]);

  const playRadarLock = useCallback((range: number) => {
    // Higher pitch as range closes (0-10000m range)
    const normalized = Math.min(range / 8000, 1);
    const freq = 400 + (1 - normalized) * 800;
    const interval = 50 + normalized * 150; // Beep faster when closer
    playTone(freq, interval / 1000 * 0.8, 'square', 0.04);
  }, [playTone]);

  const playUIClick = useCallback(() => {
    playTone(880, 0.04, 'sine', 0.03);
  }, [playTone]);

  const playAlert = useCallback(() => {
    // Two-tone alert
    playTone(600, 0.15, 'square', 0.05);
    setTimeout(() => playTone(800, 0.15, 'square', 0.05), 160);
  }, [playTone]);

  const startAmbient = useCallback(() => {
    if (muted || ambientRef.current) return;
    try {
      const ctx = getContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.value = 42;
      gain.gain.value = 0.015;

      osc.connect(gain).connect(ctx.destination);
      osc.start();

      ambientRef.current = osc;
      ambientGainRef.current = gain;
    } catch {
      // Audio context may not be available
    }
  }, [muted, getContext]);

  const stopAmbient = useCallback(() => {
    if (ambientRef.current) {
      try {
        ambientRef.current.stop();
      } catch {
        // Already stopped
      }
      ambientRef.current = null;
      ambientGainRef.current = null;
    }
  }, []);

  const toggleMute = useCallback(() => {
    setMuted(prev => {
      if (!prev) {
        stopAmbient();
      }
      return !prev;
    });
  }, [stopAmbient]);

  return {
    playLaunch,
    playExplosion,
    playRadarLock,
    playUIClick,
    playAlert,
    startAmbient,
    stopAmbient,
    muted,
    toggleMute,
  };
}
