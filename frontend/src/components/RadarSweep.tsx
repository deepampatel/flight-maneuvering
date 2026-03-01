/**
 * RadarSweep - PPI (Plan Position Indicator) Radar Display
 *
 * A circular radar overlay rendered as a picture-in-picture element
 * in the top-left corner. Shows a rotating sweep line with green
 * phosphor CRT aesthetic, range rings, threat blips with afterglow,
 * battery markers, and a north indicator.
 *
 * Phase 7 — Tactical Display
 */

import { useMemo, useState, useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import type { SimStateEvent, EntityState, BatteryState, Vec3 } from '../types';

interface RadarSweepProps {
  state: SimStateEvent | null;
  maxRange?: number; // defaults to 15000 (15km), in meters
}

/** Unique ID for the CSS keyframes injection (only once per mount). */
const SWEEP_KEYFRAMES = `
@keyframes radarSweepRotate {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
`;

/** Convert world position (meters) to radar-space pixel coords. */
function worldToRadar(
  pos: Vec3,
  center: Vec3,
  scale: number,
  radarRadius: number,
  radarCenter: number,
): { x: number; y: number } {
  const dx = (pos.x - center.x) * scale;
  const dy = (pos.y - center.y) * scale;
  return {
    x: radarCenter + dx * radarRadius,
    y: radarCenter - dy * radarRadius, // y is inverted (screen coords)
  };
}

/** Determine blip colour from entity state. */
function blipColor(entity: EntityState): string {
  if (entity.type === 'interceptor') return '#4488ff';
  switch (entity.engagement_decision) {
    case 'engage':
      return '#ff3333';
    case 'track_only':
      return '#ffcc00';
    case 'ignore':
      return '#888888';
    default:
      return '#888888';
  }
}

/** After-glow: track blip positions keyed by entity ID with a timestamp. */
interface BlipTrail {
  x: number;
  y: number;
  color: string;
  stamp: number; // performance.now()
}

export function RadarSweep({ state, maxRange = 15000 }: RadarSweepProps) {
  const SIZE = 200;
  const RADIUS = 85; // usable radar circle radius (px)
  const CENTER = SIZE / 2;

  // Inject keyframes once
  const styleInjected = useRef(false);
  useEffect(() => {
    if (styleInjected.current) return;
    const style = document.createElement('style');
    style.textContent = SWEEP_KEYFRAMES;
    document.head.appendChild(style);
    styleInjected.current = true;
    return () => {
      document.head.removeChild(style);
      styleInjected.current = false;
    };
  }, []);

  // Trail state for afterglow
  const [trails, setTrails] = useState<BlipTrail[]>([]);
  const trailsRef = useRef<BlipTrail[]>([]);

  // Compute center-of-mass and scale from all positions
  const { worldCenter, scale } = useMemo(() => {
    const entities = state?.entities || [];
    const batteries = state?.batteries || [];

    const allPositions: Vec3[] = [
      ...entities.map((e) => e.position),
      ...batteries.map((b) => b.position),
    ];

    if (allPositions.length === 0) {
      return { worldCenter: { x: 0, y: 0, z: 0 } as Vec3, scale: 1 / maxRange };
    }

    // Compute midpoint
    let cx = 0,
      cy = 0,
      cz = 0;
    for (const p of allPositions) {
      cx += p.x;
      cy += p.y;
      cz += p.z;
    }
    const n = allPositions.length;
    const wc: Vec3 = { x: cx / n, y: cy / n, z: cz / n };

    // Find max distance from center to determine scale
    let maxDist = 0;
    for (const p of allPositions) {
      const dx = p.x - wc.x;
      const dy = p.y - wc.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > maxDist) maxDist = d;
    }

    // Use the larger of entity spread or maxRange for scale, with 10% padding
    const effectiveRange = Math.max(maxDist * 1.1, maxRange);
    return { worldCenter: wc, scale: 1 / effectiveRange };
  }, [state?.entities, state?.batteries, maxRange]);

  // Compute blip positions
  const blips = useMemo(() => {
    const entities = state?.entities || [];
    return entities.map((e) => {
      const pos = worldToRadar(e.position, worldCenter, scale, RADIUS, CENTER);
      // Clamp to circle
      const dx = pos.x - CENTER;
      const dy = pos.y - CENTER;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > RADIUS) {
        const f = RADIUS / dist;
        pos.x = CENTER + dx * f;
        pos.y = CENTER + dy * f;
      }
      return { id: e.id, ...pos, color: blipColor(e), type: e.type };
    });
  }, [state?.entities, worldCenter, scale, RADIUS, CENTER]);

  // Battery markers
  const batteryMarkers = useMemo(() => {
    const batteries = state?.batteries || [];
    return batteries.map((b: BatteryState) => {
      const pos = worldToRadar(b.position, worldCenter, scale, RADIUS, CENTER);
      const dx = pos.x - CENTER;
      const dy = pos.y - CENTER;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > RADIUS) {
        const f = RADIUS / dist;
        pos.x = CENTER + dx * f;
        pos.y = CENTER + dy * f;
      }
      return { id: b.id, ...pos, name: b.name };
    });
  }, [state?.batteries, worldCenter, scale, RADIUS, CENTER]);

  // Update trails for afterglow effect
  useEffect(() => {
    const now = performance.now();
    const newTrails: BlipTrail[] = blips.map((b) => ({
      x: b.x,
      y: b.y,
      color: b.color,
      stamp: now,
    }));

    // Merge with existing, remove stale (> 2s)
    const merged = [
      ...trailsRef.current.filter((t) => now - t.stamp < 2000),
      ...newTrails,
    ];
    trailsRef.current = merged;
    setTrails(merged);
  }, [blips]);

  // ---- Styles ----

  const containerStyle: CSSProperties = {
    position: 'absolute',
    top: 12,
    left: 12,
    width: SIZE,
    height: SIZE + 22, // extra for range label
    zIndex: 100,
    pointerEvents: 'none',
    fontFamily: "'Courier New', monospace",
  };

  const radarCircleStyle: CSSProperties = {
    position: 'relative',
    width: SIZE,
    height: SIZE,
    borderRadius: '50%',
    background: 'radial-gradient(circle, #0a1a0a 0%, #050e05 70%, #020802 100%)',
    border: '2px solid #1a3a1a',
    overflow: 'hidden',
    boxShadow: '0 0 12px rgba(0, 255, 0, 0.15), inset 0 0 30px rgba(0, 20, 0, 0.8)',
  };

  const sweepStyle: CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    animation: 'radarSweepRotate 4s linear infinite',
    transformOrigin: '50% 50%',
    pointerEvents: 'none',
  };

  const rangeLabelStyle: CSSProperties = {
    textAlign: 'center',
    color: '#338833',
    fontSize: 9,
    marginTop: 4,
    letterSpacing: 1,
  };

  const northStyle: CSSProperties = {
    position: 'absolute',
    top: 4,
    left: '50%',
    transform: 'translateX(-50%)',
    color: '#44cc44',
    fontSize: 10,
    fontWeight: 'bold',
    zIndex: 10,
    textShadow: '0 0 4px #00ff00',
  };

  // Range rings
  const ringRadii = [0.25, 0.5, 0.75, 1.0];

  return (
    <div style={containerStyle}>
      <div style={radarCircleStyle}>
        {/* North indicator */}
        <div style={northStyle}>N</div>

        {/* Cross-hairs */}
        <div
          style={{
            position: 'absolute',
            top: CENTER,
            left: CENTER - RADIUS,
            width: RADIUS * 2,
            height: 1,
            background: 'rgba(0, 100, 0, 0.25)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: CENTER - RADIUS,
            left: CENTER,
            width: 1,
            height: RADIUS * 2,
            background: 'rgba(0, 100, 0, 0.25)',
          }}
        />

        {/* Range rings */}
        {ringRadii.map((frac) => {
          const r = RADIUS * frac;
          return (
            <div
              key={frac}
              style={{
                position: 'absolute',
                top: CENTER - r,
                left: CENTER - r,
                width: r * 2,
                height: r * 2,
                borderRadius: '50%',
                border: '1px solid rgba(0, 100, 0, 0.3)',
                pointerEvents: 'none',
              }}
            />
          );
        })}

        {/* Sweep line — drawn as a conic gradient */}
        <div style={sweepStyle}>
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              borderRadius: '50%',
              background:
                'conic-gradient(from 0deg, rgba(0,255,0,0.12) 0deg, rgba(0,255,0,0) 30deg, transparent 30deg)',
            }}
          />
          {/* Bright leading edge */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: '50%',
              width: 1,
              height: '50%',
              background: 'linear-gradient(to top, rgba(0,255,0,0.05), rgba(0,255,0,0.7))',
              transformOrigin: 'bottom center',
            }}
          />
        </div>

        {/* Afterglow trails */}
        {trails.map((trail, i) => {
          const age = (performance.now() - trail.stamp) / 2000; // 0..1 over 2s
          const opacity = Math.max(0, 1 - age) * 0.5;
          if (opacity <= 0) return null;
          return (
            <div
              key={`trail-${i}`}
              style={{
                position: 'absolute',
                top: trail.y - 2,
                left: trail.x - 2,
                width: 4,
                height: 4,
                borderRadius: '50%',
                background: trail.color,
                opacity,
                transition: 'opacity 0.5s ease-out',
                pointerEvents: 'none',
              }}
            />
          );
        })}

        {/* Battery markers — green hexagons (approximated as rotated squares) */}
        {batteryMarkers.map((bm) => (
          <div
            key={`bat-${bm.id}`}
            style={{
              position: 'absolute',
              top: bm.y - 5,
              left: bm.x - 5,
              width: 10,
              height: 10,
              background: 'rgba(0, 200, 0, 0.7)',
              clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)',
              zIndex: 5,
              pointerEvents: 'none',
            }}
            title={bm.name}
          />
        ))}

        {/* Entity blips */}
        {blips.map((blip) => (
          <div
            key={`blip-${blip.id}`}
            style={{
              position: 'absolute',
              top: blip.y - 3,
              left: blip.x - 3,
              width: 6,
              height: 6,
              borderRadius: blip.type === 'interceptor' ? '50%' : '1px',
              background: blip.color,
              boxShadow: `0 0 4px ${blip.color}`,
              zIndex: 6,
              pointerEvents: 'none',
            }}
            title={blip.id}
          />
        ))}
      </div>

      {/* Range label */}
      <div style={rangeLabelStyle}>
        RANGE {(maxRange / 1000).toFixed(0)} KM
      </div>
    </div>
  );
}
