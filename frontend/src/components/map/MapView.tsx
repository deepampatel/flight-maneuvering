/**
 * MapView — Google Maps 2D tactical view for the simulation.
 *
 * Replaces the Three.js overhead view when viewMode === 'map'.
 * Shows a dark Google Maps base layer with entity markers, trails,
 * range circles, and protected area overlays.
 */

import { APIProvider, Map } from '@vis.gl/react-google-maps';
import type { SimStateEvent, SimStateEventWithEnvironment, InterceptGeometry, BuilderBatteryConfig, BuilderProtectedArea } from '../../types';
import type { GlobeConfig } from '../../utils/globeCoords';
import { MapEntityMarkers } from './MapEntityMarkers';
import { MapOverlays } from './MapOverlays';
import { MapAutoFit } from './MapAutoFit';

interface MapViewProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  interceptGeometry?: InterceptGeometry[] | null;
  globeConfig: GlobeConfig;
  selectedEntityId?: string | null;
  onSelectEntity?: (id: string | null) => void;
  previewBatteries?: BuilderBatteryConfig[];
  previewProtectedAreas?: BuilderProtectedArea[];
}

export function MapView({
  state,
  interceptGeometry,
  globeConfig,
  selectedEntityId,
  onSelectEntity,
  previewBatteries,
  previewProtectedAreas,
}: MapViewProps) {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string || '';

  if (!apiKey) {
    return (
      <div className="map-no-key">
        <div className="map-no-key-content">
          <div className="map-no-key-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
          </div>
          <p className="map-no-key-title">Google Maps API Key Required</p>
          <p className="map-no-key-text">
            Set <code>VITE_GOOGLE_MAPS_API_KEY</code> in <code>frontend/.env.local</code>
          </p>
        </div>
      </div>
    );
  }

  const center = { lat: globeConfig.originLat, lng: globeConfig.originLng };

  return (
    <div className="map-view-container">
      <APIProvider apiKey={apiKey} libraries={['geometry']}>
        <Map
          defaultCenter={center}
          defaultZoom={11}
          colorScheme="DARK"
          disableDefaultUI={true}
          zoomControl={true}
          gestureHandling="greedy"
          mapTypeId="roadmap"
          style={{ width: '100%', height: '100%' }}
          mapId="intercept-tactical-map"
        >
          <MapEntityMarkers
            state={state}
            globeConfig={globeConfig}
            selectedEntityId={selectedEntityId}
            onSelectEntity={onSelectEntity}
            previewBatteries={previewBatteries}
          />
          <MapOverlays
            state={state}
            globeConfig={globeConfig}
            interceptGeometry={interceptGeometry}
            previewProtectedAreas={previewProtectedAreas}
          />
          <MapAutoFit
            state={state}
            globeConfig={globeConfig}
            previewBatteries={previewBatteries}
            previewProtectedAreas={previewProtectedAreas}
          />
        </Map>
      </APIProvider>
    </div>
  );
}
