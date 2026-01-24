/**
 * LaunchEventToast - Command Center Alert System
 *
 * Displays real-time alerts for launcher events:
 * - Target detection
 * - Missile launches
 * - Intercept results
 * - Magazine status
 *
 * Creates an immersive command center experience with visual/audio feedback.
 */

import { useEffect, useState, useCallback } from 'react';
import type { LaunchEvent, LaunchEventType, LauncherState } from '../types';

interface LaunchEventToastProps {
  launchers: LauncherState[] | null;
  enabled: boolean;
}

interface TrackedState {
  launcherMissiles: Record<string, number>;
  trackedTargets: Record<string, Set<string>>;
  engagedTargets: Record<string, Set<string>>;
}

// Generate unique event ID
let eventCounter = 0;
const generateEventId = () => `evt_${Date.now()}_${++eventCounter}`;

// Event display configuration
const EVENT_CONFIG: Record<LaunchEventType, { icon: string; color: string; duration: number }> = {
  target_detected: { icon: '📡', color: '#fbbf24', duration: 4000 },
  missile_launched: { icon: '🚀', color: '#ef4444', duration: 5000 },
  magazine_depleted: { icon: '⚠️', color: '#f97316', duration: 6000 },
  intercept_success: { icon: '💥', color: '#22c55e', duration: 6000 },
  intercept_miss: { icon: '❌', color: '#ef4444', duration: 5000 },
  target_lost: { icon: '👁️', color: '#94a3b8', duration: 3000 },
};

export function LaunchEventToast({ launchers, enabled }: LaunchEventToastProps) {
  const [events, setEvents] = useState<LaunchEvent[]>([]);
  const [trackedState, setTrackedState] = useState<TrackedState>({
    launcherMissiles: {},
    trackedTargets: {},
    engagedTargets: {},
  });

  // Add a new event
  const addEvent = useCallback((event: Omit<LaunchEvent, 'id' | 'timestamp'>) => {
    const newEvent: LaunchEvent = {
      ...event,
      id: generateEventId(),
      timestamp: Date.now(),
    };

    setEvents((prev) => {
      // Limit to 5 visible events
      const updated = [newEvent, ...prev].slice(0, 5);
      return updated;
    });

    // Auto-remove after duration
    const config = EVENT_CONFIG[event.type];
    setTimeout(() => {
      setEvents((prev) => prev.filter((e) => e.id !== newEvent.id));
    }, config.duration);
  }, []);

  // Detect changes in launcher state and generate events
  useEffect(() => {
    if (!enabled || !launchers) return;

    launchers.forEach((launcher) => {
      const prevMissiles = trackedState.launcherMissiles[launcher.id];
      const prevTracked = trackedState.trackedTargets[launcher.id] || new Set<string>();
      const prevEngaged = trackedState.engagedTargets[launcher.id] || new Set<string>();

      const currentTracked = new Set(launcher.tracked_targets.map((t) => t.target_id));
      const currentEngaged = new Set(launcher.engaged_targets);

      // Detect new target acquisitions
      currentTracked.forEach((targetId) => {
        if (!prevTracked.has(targetId)) {
          const track = launcher.tracked_targets.find((t) => t.target_id === targetId);
          addEvent({
            type: 'target_detected',
            launcher_id: launcher.id,
            target_id: targetId,
            message: `${launcher.id} acquired ${targetId}`,
            details: {
              range: track?.range,
              bearing: track?.bearing,
            },
          });
        }
      });

      // Detect lost targets
      prevTracked.forEach((targetId) => {
        if (!currentTracked.has(targetId) && !prevEngaged.has(targetId)) {
          addEvent({
            type: 'target_lost',
            launcher_id: launcher.id,
            target_id: targetId,
            message: `${launcher.id} lost track: ${targetId}`,
          });
        }
      });

      // Detect missile launches (when missiles_remaining decreases)
      if (prevMissiles !== undefined && launcher.missiles_remaining < prevMissiles) {
        // Find newly engaged targets
        currentEngaged.forEach((targetId) => {
          if (!prevEngaged.has(targetId)) {
            const track = launcher.tracked_targets.find((t) => t.target_id === targetId);
            addEvent({
              type: 'missile_launched',
              launcher_id: launcher.id,
              target_id: targetId,
              interceptor_id: track?.assigned_interceptor || undefined,
              message: `${launcher.id} launched at ${targetId}`,
              details: {
                range: track?.range,
                missiles_remaining: launcher.missiles_remaining,
              },
            });
          }
        });

        // Check for magazine depletion
        if (launcher.missiles_remaining === 0) {
          addEvent({
            type: 'magazine_depleted',
            launcher_id: launcher.id,
            message: `${launcher.id} WINCHESTER - No missiles remaining`,
            details: {
              missiles_remaining: 0,
            },
          });
        }
      }
    });

    // Update tracked state
    setTrackedState({
      launcherMissiles: launchers.reduce(
        (acc, l) => ({ ...acc, [l.id]: l.missiles_remaining }),
        {}
      ),
      trackedTargets: launchers.reduce(
        (acc, l) => ({
          ...acc,
          [l.id]: new Set(l.tracked_targets.map((t) => t.target_id)),
        }),
        {}
      ),
      engagedTargets: launchers.reduce(
        (acc, l) => ({ ...acc, [l.id]: new Set(l.engaged_targets) }),
        {}
      ),
    });
  }, [launchers, enabled, addEvent]);

  // Clear events when disabled
  useEffect(() => {
    if (!enabled) {
      setEvents([]);
      setTrackedState({
        launcherMissiles: {},
        trackedTargets: {},
        engagedTargets: {},
      });
    }
  }, [enabled]);

  if (!enabled || events.length === 0) {
    return null;
  }

  return (
    <div className="launch-events-container">
      {events.map((event, index) => {
        const config = EVENT_CONFIG[event.type];
        const age = (Date.now() - event.timestamp) / config.duration;
        const opacity = Math.max(0.4, 1 - age * 0.6);

        return (
          <div
            key={event.id}
            className={`launch-event ${event.type}`}
            style={{
              '--event-color': config.color,
              opacity,
              transform: `translateY(${index * 4}px)`,
              zIndex: 1000 - index,
            } as React.CSSProperties}
          >
            <div className="event-icon">{config.icon}</div>
            <div className="event-content">
              <div className="event-header">
                <span className="event-type">{formatEventType(event.type)}</span>
                <span className="event-time">{formatTime(event.timestamp)}</span>
              </div>
              <div className="event-message">{event.message}</div>
              {event.details && (
                <div className="event-details">
                  {event.details.range && (
                    <span className="detail-item">
                      RNG: {(event.details.range / 1000).toFixed(1)}km
                    </span>
                  )}
                  {event.details.bearing !== undefined && (
                    <span className="detail-item">
                      BRG: {event.details.bearing.toFixed(0)}°
                    </span>
                  )}
                  {event.details.missiles_remaining !== undefined && (
                    <span className="detail-item mag">
                      MAG: {event.details.missiles_remaining}
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="event-progress">
              <div
                className="event-progress-bar"
                style={{
                  width: `${Math.max(0, (1 - age) * 100)}%`,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Format event type for display
function formatEventType(type: LaunchEventType): string {
  const typeMap: Record<LaunchEventType, string> = {
    target_detected: 'TGT ACQUIRED',
    missile_launched: 'FOX THREE',
    magazine_depleted: 'WINCHESTER',
    intercept_success: 'SPLASH',
    intercept_miss: 'MISS',
    target_lost: 'TGT LOST',
  };
  return typeMap[type] || type.replace(/_/g, ' ').toUpperCase();
}

// Format timestamp for display
function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}
