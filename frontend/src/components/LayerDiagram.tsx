/**
 * LayerDiagram - Multi-Layer Defense Visualization
 *
 * Nested semicircles showing Iron Dome, David's Sling, and Arrow
 * defense layers with battery status per tier.
 *
 * Phase 8 — Multi-Layer Defense
 */

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import type { SimStateEvent, BatteryState } from '../types';

interface LayerDiagramProps {
  state: SimStateEvent | null;
}

interface TierInfo {
  name: string;
  tier: string;
  rangeLabel: string;
  color: string;
  glowColor: string;
  batteries: BatteryState[];
  totalAmmo: number;
  maxAmmo: number;
  activeEngagements: number;
}

export function LayerDiagram({ state }: LayerDiagramProps) {
  const WIDTH = 220;
  const HEIGHT = 140;

  const tiers = useMemo<TierInfo[]>(() => {
    const batteries = state?.batteries || [];

    const tierMap: Record<string, BatteryState[]> = {
      iron_dome: [],
      davids_sling: [],
      arrow: [],
    };

    for (const bat of batteries) {
      const tier = bat.tier || 'iron_dome';
      if (tierMap[tier]) {
        tierMap[tier].push(bat);
      }
    }

    return [
      {
        name: 'ARROW',
        tier: 'arrow',
        rangeLabel: '100-2400km',
        color: '#a855f7',
        glowColor: 'rgba(168, 85, 247, 0.3)',
        batteries: tierMap.arrow,
        totalAmmo: tierMap.arrow.reduce((s, b) => s + b.missiles_remaining, 0),
        maxAmmo: tierMap.arrow.reduce((s, b) => s + b.missiles_total, 0),
        activeEngagements: tierMap.arrow.reduce((s, b) => s + b.active_engagements, 0),
      },
      {
        name: "DAVID'S SLING",
        tier: 'davids_sling',
        rangeLabel: '40-300km',
        color: '#22d3ee',
        glowColor: 'rgba(34, 211, 238, 0.3)',
        batteries: tierMap.davids_sling,
        totalAmmo: tierMap.davids_sling.reduce((s, b) => s + b.missiles_remaining, 0),
        maxAmmo: tierMap.davids_sling.reduce((s, b) => s + b.missiles_total, 0),
        activeEngagements: tierMap.davids_sling.reduce((s, b) => s + b.active_engagements, 0),
      },
      {
        name: 'IRON DOME',
        tier: 'iron_dome',
        rangeLabel: '4-70km',
        color: '#60a5fa',
        glowColor: 'rgba(96, 165, 250, 0.3)',
        batteries: tierMap.iron_dome,
        totalAmmo: tierMap.iron_dome.reduce((s, b) => s + b.missiles_remaining, 0),
        maxAmmo: tierMap.iron_dome.reduce((s, b) => s + b.missiles_total, 0),
        activeEngagements: tierMap.iron_dome.reduce((s, b) => s + b.active_engagements, 0),
      },
    ];
  }, [state?.batteries]);

  // Don't render if no multi-layer batteries
  const hasMultiLayer = tiers.some((t) => t.batteries.length > 0 && t.tier !== 'iron_dome') ||
    tiers.filter((t) => t.batteries.length > 0).length > 1;
  if (!hasMultiLayer) return null;

  const containerStyle: CSSProperties = {
    position: 'absolute',
    top: 12,
    right: 12,
    width: WIDTH,
    zIndex: 100,
    pointerEvents: 'none',
    fontFamily: "'Courier New', monospace",
  };

  const panelStyle: CSSProperties = {
    background: 'rgba(5, 10, 20, 0.9)',
    border: '1px solid #1a2a3a',
    borderRadius: 2,
    padding: '6px 8px',
  };

  const titleStyle: CSSProperties = {
    fontSize: 9,
    color: '#8899aa',
    letterSpacing: 1.5,
    marginBottom: 6,
    textAlign: 'center',
  };

  // SVG semicircle diagram
  const arcRadii = [35, 55, 75]; // inner to outer

  return (
    <div style={containerStyle}>
      <div style={panelStyle}>
        <div style={titleStyle}>DEFENSE LAYERS</div>

        {/* SVG semicircle arcs */}
        <svg width={WIDTH - 16} height={HEIGHT - 30} viewBox={`0 0 ${WIDTH - 16} ${HEIGHT - 30}`}>
          {tiers.map((tier, i) => {
            const radius = arcRadii[tiers.length - 1 - i]; // Arrow is outermost
            const hasBatteries = tier.batteries.length > 0;
            const cx = (WIDTH - 16) / 2;
            const cy = HEIGHT - 35;

            return (
              <g key={tier.tier}>
                {/* Arc */}
                <path
                  d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
                  fill="none"
                  stroke={hasBatteries ? tier.color : '#333'}
                  strokeWidth={hasBatteries ? 2 : 1}
                  strokeDasharray={hasBatteries ? 'none' : '4,3'}
                  opacity={hasBatteries ? 0.8 : 0.3}
                />
                {/* Glow effect */}
                {hasBatteries && (
                  <path
                    d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
                    fill="none"
                    stroke={tier.color}
                    strokeWidth={6}
                    opacity={0.1}
                    filter="blur(3px)"
                  />
                )}
                {/* Label */}
                <text
                  x={cx}
                  y={cy - radius + 10}
                  textAnchor="middle"
                  fill={hasBatteries ? tier.color : '#555'}
                  fontSize={7}
                  fontFamily="'Courier New', monospace"
                  opacity={hasBatteries ? 1 : 0.5}
                >
                  {tier.name}
                </text>
              </g>
            );
          })}

          {/* Center point (defended area) */}
          <circle cx={(WIDTH - 16) / 2} cy={HEIGHT - 35} r={3} fill="#4ade80" opacity={0.8} />
          <text
            x={(WIDTH - 16) / 2}
            y={HEIGHT - 25}
            textAnchor="middle"
            fill="#4ade80"
            fontSize={6}
            fontFamily="'Courier New', monospace"
          >
            DEFENDED
          </text>
        </svg>

        {/* Tier status rows */}
        {tiers.map((tier) => {
          if (tier.batteries.length === 0) return null;
          const ammoFrac = tier.maxAmmo > 0 ? tier.totalAmmo / tier.maxAmmo : 0;

          const rowStyle: CSSProperties = {
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            marginTop: 3,
            fontSize: 8,
          };

          const dotStyle: CSSProperties = {
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: tier.color,
            boxShadow: `0 0 4px ${tier.glowColor}`,
            flexShrink: 0,
          };

          const ammoBarBg: CSSProperties = {
            flex: 1,
            height: 4,
            background: '#1a1a2a',
            borderRadius: 2,
            overflow: 'hidden',
          };

          const ammoBarFill: CSSProperties = {
            width: `${ammoFrac * 100}%`,
            height: '100%',
            background: ammoFrac > 0.3 ? tier.color : '#ef4444',
            borderRadius: 2,
            transition: 'width 0.2s ease',
          };

          return (
            <div key={tier.tier} style={rowStyle}>
              <div style={dotStyle} />
              <span style={{ color: tier.color, width: 30, flexShrink: 0 }}>
                {tier.batteries.length}B
              </span>
              <div style={ammoBarBg}>
                <div style={ammoBarFill} />
              </div>
              <span style={{ color: '#8899aa', width: 28, textAlign: 'right', flexShrink: 0 }}>
                {tier.totalAmmo}
              </span>
              {tier.activeEngagements > 0 && (
                <span style={{ color: '#ff8800', flexShrink: 0 }}>
                  ⚡{tier.activeEngagements}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
