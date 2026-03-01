/**
 * EngagementTimeline - Horizontal Engagement Event Timeline
 *
 * A slim horizontal bar rendered at the bottom of the screen showing
 * a timeline of engagement events: wave spawns, intercepts, battery
 * launches, and the current time indicator. Styled with a dark C2
 * military aesthetic.
 *
 * Phase 7 — Tactical Display
 */

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import type { SimStateEvent } from '../types';

interface EngagementTimelineProps {
  state: SimStateEvent | null;
  maxTime?: number; // defaults to 60
}

/** Time labels to render along the axis. */
function makeTimeLabels(maxTime: number): number[] {
  // Generate labels at 25% intervals, always including 0 and maxTime
  const step = maxTime / 4;
  const labels: number[] = [];
  for (let t = 0; t <= maxTime; t += step) {
    labels.push(Math.round(t));
  }
  if (labels[labels.length - 1] !== maxTime) {
    labels.push(maxTime);
  }
  return labels;
}

interface TimelineMarker {
  time: number;
  color: string;
  shape: 'triangle' | 'circle' | 'square';
  label: string;
}

export function EngagementTimeline({ state, maxTime = 60 }: EngagementTimelineProps) {
  const simTime = state?.sim_time ?? 0;
  const progress = Math.min(simTime / maxTime, 1);

  // Collect markers from state
  const markers = useMemo<TimelineMarker[]>(() => {
    if (!state) return [];
    const result: TimelineMarker[] = [];

    // Intercept events — from intercepted_pairs
    // We don't have exact timestamps per pair, so we mark them at current sim_time
    // when they first appear. For a richer timeline, the backend would need to
    // provide per-event timestamps. Here we approximate using engagement_log.
    const seenIntercepts = new Set<string>();

    // Battery engagement log entries — orange squares for launches, green circles for intercepts
    if (state.batteries) {
      for (const battery of state.batteries) {
        if (battery.engagement_log) {
          for (const entry of battery.engagement_log) {
            if (entry.type === 'launch' || entry.type === 'missile_launched') {
              result.push({
                time: entry.time,
                color: '#ff8800',
                shape: 'square',
                label: `Launch: ${battery.name} -> ${entry.threat_id}`,
              });
            }
            if (entry.type === 'intercept' || entry.type === 'intercept_success') {
              const key = `${entry.threat_id}-${entry.interceptor_id}`;
              if (!seenIntercepts.has(key)) {
                seenIntercepts.add(key);
                result.push({
                  time: entry.time,
                  color: '#33cc33',
                  shape: 'circle',
                  label: `Intercept: ${entry.interceptor_id} x ${entry.threat_id}`,
                });
              }
            }
          }
        }
      }
    }

    // Wave spawn markers — red triangles
    // If wave_info is present and waves have spawned, mark estimated spawn times
    if (state.wave_info && state.engagement_stats) {
      const wavesSpawned = state.engagement_stats.waves_spawned || 0;
      if (wavesSpawned > 0 && maxTime > 0) {
        // Distribute wave markers evenly across elapsed time as an approximation
        const interval = simTime / wavesSpawned;
        for (let w = 0; w < wavesSpawned; w++) {
          result.push({
            time: interval * w,
            color: '#ff3333',
            shape: 'triangle',
            label: `Wave ${w + 1}`,
          });
        }
      }
    }

    return result;
  }, [state, maxTime]);

  const timeLabels = useMemo(() => makeTimeLabels(maxTime), [maxTime]);

  // ---- Styles ----

  const barContainer: CSSProperties = {
    position: 'relative',
    width: '100%',
    height: 30,
    background: '#18181b',
    borderTop: '1px solid rgba(63, 63, 70, 0.5)',
    fontFamily: "'Inter', system-ui, sans-serif",
    fontSize: 9,
    color: '#71717a',
    userSelect: 'none',
    overflow: 'hidden',
  };

  const trackStyle: CSSProperties = {
    position: 'absolute',
    top: 12,
    left: 40,
    right: 40,
    height: 3,
    background: '#27272a',
    borderRadius: 2,
  };

  const progressStyle: CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: `${progress * 100}%`,
    height: '100%',
    background: 'linear-gradient(to right, #1e3a5f, #3b82f6)',
    borderRadius: 2,
    transition: 'width 0.1s linear',
  };

  // Compute track geometry for positioning
  const TRACK_LEFT = 40; // px
  const TRACK_RIGHT = 40; // px

  /** Convert time to percentage along the track (0..1). */
  const timeFrac = (t: number) => Math.min(Math.max(t / maxTime, 0), 1);

  // Shape renderers
  const renderMarker = (marker: TimelineMarker, index: number) => {
    const frac = timeFrac(marker.time);
    const baseStyle: CSSProperties = {
      position: 'absolute',
      top: 6,
      left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT + TRACK_RIGHT}px) * ${frac})`,
      zIndex: 5,
      pointerEvents: 'none',
      transform: 'translateX(-50%)',
    };

    if (marker.shape === 'triangle') {
      return (
        <div key={`marker-${index}`} style={baseStyle} title={marker.label}>
          <div
            style={{
              width: 0,
              height: 0,
              borderLeft: '4px solid transparent',
              borderRight: '4px solid transparent',
              borderBottom: `7px solid ${marker.color}`,
              opacity: 0.9,
            }}
          />
        </div>
      );
    }

    if (marker.shape === 'circle') {
      return (
        <div key={`marker-${index}`} style={baseStyle} title={marker.label}>
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: marker.color,
              opacity: 0.9,
              marginTop: 3,
            }}
          />
        </div>
      );
    }

    // square
    return (
      <div key={`marker-${index}`} style={baseStyle} title={marker.label}>
        <div
          style={{
            width: 6,
            height: 6,
            background: marker.color,
            boxShadow: `0 0 3px ${marker.color}`,
            marginTop: 3,
          }}
        />
      </div>
    );
  };

  return (
    <div style={barContainer}>
      {/* Time labels */}
      {timeLabels.map((t) => {
        const frac = timeFrac(t);
        return (
          <div
            key={`label-${t}`}
            style={{
              position: 'absolute',
              bottom: 1,
              left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT + TRACK_RIGHT}px) * ${frac})`,
              transform: 'translateX(-50%)',
              color: '#71717a',
              fontSize: 8,
              letterSpacing: 0.5,
            }}
          >
            {t}s
          </div>
        );
      })}

      {/* Track bar */}
      <div style={trackStyle}>
        <div style={progressStyle} />
      </div>

      {/* Current time indicator */}
      <div
        style={{
          position: 'absolute',
          top: 4,
          left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT + TRACK_RIGHT}px) * ${progress})`,
          height: 22,
          width: 1,
          background: '#fafafa',
          opacity: 0.8,
          zIndex: 10,
          transition: 'left 0.1s linear',
          transform: 'translateX(-0.5px)',
        }}
      />

      {/* Current time readout */}
      <div
        style={{
          position: 'absolute',
          top: 1,
          left: `calc(${TRACK_LEFT}px + (100% - ${TRACK_LEFT + TRACK_RIGHT}px) * ${progress} + 4px)`,
          color: '#88cc88',
          fontSize: 8,
          fontWeight: 'bold',
          zIndex: 11,
          whiteSpace: 'nowrap',
          transition: 'left 0.1s linear',
        }}
      >
        T+{simTime.toFixed(1)}s
      </div>

      {/* Event markers */}
      {markers.map((m, i) => renderMarker(m, i))}
    </div>
  );
}
