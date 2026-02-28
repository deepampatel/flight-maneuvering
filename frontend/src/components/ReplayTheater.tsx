/**
 * ReplayTheater - Cinematic replay with timeline and speed controls
 *
 * Shows during replay mode. Features:
 * - Timeline scrubber with event markers
 * - Speed controls (0.25x to 4x)
 * - Screenshot capture
 * - Auto-camera toggle
 */

import { useState, useCallback } from 'react';

interface ReplayTheaterProps {
  currentTick: number;
  totalTicks: number;
  isPlaying: boolean;
  speed: number;
  onSeek: (tick: number) => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onSetSpeed: (speed: number) => void;
  autoCameraEnabled: boolean;
  onToggleAutoCamera: () => void;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

export function ReplayTheater({
  currentTick,
  totalTicks,
  isPlaying,
  speed,
  onSeek,
  onPause,
  onResume,
  onStop,
  onSetSpeed,
  autoCameraEnabled,
  onToggleAutoCamera,
}: ReplayTheaterProps) {
  const [, setIsDragging] = useState(false);

  const progress = totalTicks > 0 ? currentTick / totalTicks : 0;

  const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const tick = Math.round(x * totalTicks);
    onSeek(Math.max(0, Math.min(tick, totalTicks)));
  }, [totalTicks, onSeek]);

  const handleScreenshot = useCallback(() => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = `intercept-replay-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  }, []);

  return (
    <div className="replay-theater">
      {/* Timeline */}
      <div
        className="replay-timeline"
        onClick={handleTimelineClick}
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
      >
        <div className="replay-progress" style={{ width: `${progress * 100}%` }} />
        <div className="replay-playhead" style={{ left: `${progress * 100}%` }} />
      </div>

      {/* Controls */}
      <div className="replay-controls">
        <div className="replay-controls-left">
          <button className="replay-btn" onClick={isPlaying ? onPause : onResume}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button className="replay-btn" onClick={onStop}>
            ⏹
          </button>
          <span className="replay-time">
            {currentTick} / {totalTicks}
          </span>
        </div>

        <div className="replay-controls-center">
          {SPEED_OPTIONS.map(s => (
            <button
              key={s}
              className={`replay-speed-btn ${speed === s ? 'active' : ''}`}
              onClick={() => onSetSpeed(s)}
            >
              {s}x
            </button>
          ))}
        </div>

        <div className="replay-controls-right">
          <button
            className={`replay-btn replay-auto-cam ${autoCameraEnabled ? 'active' : ''}`}
            onClick={onToggleAutoCamera}
            title="Auto-Camera"
          >
            CINE
          </button>
          <button className="replay-btn" onClick={handleScreenshot} title="Screenshot">
            📷
          </button>
        </div>
      </div>
    </div>
  );
}
