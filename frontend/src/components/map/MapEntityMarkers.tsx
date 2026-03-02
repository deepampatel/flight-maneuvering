/**
 * MapEntityMarkers — Google Maps AdvancedMarker overlays for simulation entities.
 *
 * Renders each entity (target, interceptor) as a rotated SVG arrow marker,
 * batteries as hexagonal badges with tier color, and launchers as square badges.
 *
 * Uses React.memo for performance — ~50 entities × ~60Hz state updates.
 */

import { memo, useMemo } from 'react';
import { AdvancedMarker } from '@vis.gl/react-google-maps';
import type { EntityState, SimStateEvent, SimStateEventWithEnvironment, LauncherState, BuilderBatteryConfig } from '../../types';
import type { GlobeConfig } from '../../utils/globeCoords';
import { enuToRealLatLng } from '../../utils/globeCoords';
import { TIER_COLORS, ENTITY_COLORS } from './mapStyles';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
interface MapEntityMarkersProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  globeConfig: GlobeConfig;
  selectedEntityId?: string | null;
  onSelectEntity?: (id: string | null) => void;
  previewBatteries?: BuilderBatteryConfig[];
}

// ---------------------------------------------------------------------------
// Individual marker components
// ---------------------------------------------------------------------------

/** Arrow marker for missiles/aircraft. Rotated by heading. */
const EntityArrow = memo(function EntityArrow({
  entity, globeConfig, isSelected, isIntercepted, onClick,
}: {
  entity: EntityState;
  globeConfig: GlobeConfig;
  isSelected: boolean;
  isIntercepted: boolean;
  onClick: () => void;
}) {
  const pos = enuToRealLatLng(entity.position, globeConfig.originLat, globeConfig.originLng);
  const heading = Math.atan2(entity.velocity.x, entity.velocity.y) * (180 / Math.PI);
  const isTarget = entity.type === 'target';
  const color = isTarget ? ENTITY_COLORS.target : ENTITY_COLORS.interceptor;

  if (isIntercepted) return null;

  return (
    <AdvancedMarker
      position={{ lat: pos.lat, lng: pos.lng }}
      onClick={onClick}
      zIndex={isSelected ? 100 : isTarget ? 50 : 40}
    >
      <div
        className={`map-marker ${isSelected ? 'selected' : ''}`}
        style={{ transform: `rotate(${heading}deg)` }}
        title={`${entity.id} — alt: ${Math.round(entity.position.z)}m`}
      >
        <svg width="20" height="20" viewBox="0 0 20 20">
          {/* Arrow pointing up */}
          <path
            d="M10 2 L16 14 L10 10 L4 14 Z"
            fill={color}
            stroke={isSelected ? '#fbbf24' : '#ffffff'}
            strokeWidth={isSelected ? 1.5 : 0.5}
            opacity={0.95}
          />
        </svg>
        {/* Entity ID label */}
        <span className="map-marker-label">{entity.id}</span>
      </div>
    </AdvancedMarker>
  );
});

/** Battery marker — hexagonal badge with tier color. */
const BatteryMarker = memo(function BatteryMarker({
  battery, globeConfig, isLive,
}: {
  battery: { id: string; name: string; position: { x: number; y: number; z: number }; tier: string; radar_range?: number; missiles_remaining?: number; missiles_total?: number };
  globeConfig: GlobeConfig;
  isLive: boolean;
}) {
  const pos = enuToRealLatLng(battery.position, globeConfig.originLat, globeConfig.originLng);
  const tierColor = TIER_COLORS[battery.tier] || ENTITY_COLORS.battery;
  const ammoText = battery.missiles_remaining !== undefined
    ? `${battery.missiles_remaining}/${battery.missiles_total}`
    : '';

  return (
    <AdvancedMarker
      position={{ lat: pos.lat, lng: pos.lng }}
      zIndex={30}
    >
      <div className="map-battery-marker" style={{ borderColor: tierColor, background: `${tierColor}22` }}>
        <svg width="12" height="12" viewBox="0 0 12 12">
          <path d="M6 0 L11 3 L11 9 L6 12 L1 9 L1 3 Z" fill={tierColor} opacity={0.6} />
        </svg>
        <span className="map-battery-label" style={{ color: tierColor }}>
          {battery.name || battery.id}
        </span>
        {isLive && ammoText && (
          <span className="map-battery-ammo">{ammoText}</span>
        )}
      </div>
    </AdvancedMarker>
  );
});

/** Launcher marker — small square badge. */
const LauncherMarker = memo(function LauncherMarker({
  launcher, globeConfig,
}: {
  launcher: LauncherState;
  globeConfig: GlobeConfig;
}) {
  const pos = enuToRealLatLng(launcher.position, globeConfig.originLat, globeConfig.originLng);

  return (
    <AdvancedMarker
      position={{ lat: pos.lat, lng: pos.lng }}
      zIndex={25}
    >
      <div className="map-launcher-marker">
        <svg width="14" height="14" viewBox="0 0 14 14">
          <rect x="1" y="1" width="12" height="12" rx="2" fill={ENTITY_COLORS.launcher} opacity={0.7} stroke="#fff" strokeWidth="0.5" />
          <circle cx="7" cy="7" r="2" fill="none" stroke="#fff" strokeWidth="0.8" />
        </svg>
        <span className="map-launcher-label">{launcher.missiles_remaining}/{launcher.missiles_total}</span>
      </div>
    </AdvancedMarker>
  );
});

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function MapEntityMarkers({
  state,
  globeConfig,
  selectedEntityId,
  onSelectEntity,
  previewBatteries,
}: MapEntityMarkersProps) {
  const interceptedIds = useMemo(() => {
    const set = new Set<string>();
    if (state?.intercepted_pairs) {
      for (const pair of state.intercepted_pairs) {
        set.add(pair[0]);
        set.add(pair[1]);
      }
    }
    return set;
  }, [state?.intercepted_pairs]);

  const entities = state?.entities || [];
  const batteries = state?.batteries || [];
  const launchers = (state as SimStateEventWithEnvironment)?.launchers || [];

  return (
    <>
      {/* Entity arrows (targets + interceptors) */}
      {entities.map(entity => (
        <EntityArrow
          key={entity.id}
          entity={entity}
          globeConfig={globeConfig}
          isSelected={selectedEntityId === entity.id}
          isIntercepted={interceptedIds.has(entity.id)}
          onClick={() => onSelectEntity?.(
            selectedEntityId === entity.id ? null : entity.id,
          )}
        />
      ))}

      {/* Live battery markers */}
      {batteries.map(battery => (
        <BatteryMarker
          key={battery.id}
          battery={battery}
          globeConfig={globeConfig}
          isLive={true}
        />
      ))}

      {/* Preview battery markers (before simulation starts) */}
      {!state && previewBatteries && previewBatteries.map(config => (
        <BatteryMarker
          key={config.id}
          battery={{
            id: config.id,
            name: config.name,
            position: config.position,
            tier: config.tier,
          }}
          globeConfig={globeConfig}
          isLive={false}
        />
      ))}

      {/* Launcher markers */}
      {launchers.map(launcher => (
        <LauncherMarker
          key={launcher.id}
          launcher={launcher}
          globeConfig={globeConfig}
        />
      ))}
    </>
  );
}
