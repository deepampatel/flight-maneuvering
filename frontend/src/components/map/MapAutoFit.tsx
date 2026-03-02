/**
 * MapAutoFit — auto-zoom Google Maps to fit all entities on simulation start.
 *
 * Fires once per simulation run (new run_id). Does not fight user pan/zoom.
 * Also fits to preview batteries/areas when no simulation is running.
 */

import { useEffect, useRef } from 'react';
import { useMap } from '@vis.gl/react-google-maps';
import type { SimStateEvent, SimStateEventWithEnvironment, BuilderBatteryConfig, BuilderProtectedArea } from '../../types';
import type { GlobeConfig } from '../../utils/globeCoords';
import { enuToRealLatLng } from '../../utils/globeCoords';

interface MapAutoFitProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  globeConfig: GlobeConfig;
  previewBatteries?: BuilderBatteryConfig[];
  previewProtectedAreas?: BuilderProtectedArea[];
}

export function MapAutoFit({ state, globeConfig, previewBatteries, previewProtectedAreas }: MapAutoFitProps) {
  const map = useMap();
  const prevRunIdRef = useRef<string | null>(null);
  const hasFittedPreviewRef = useRef(false);

  // Auto-fit on simulation start (new run_id)
  useEffect(() => {
    if (!map || !state?.entities?.length) return;

    if (state.run_id === prevRunIdRef.current) return;
    prevRunIdRef.current = state.run_id;

    const bounds = new google.maps.LatLngBounds();

    for (const entity of state.entities) {
      const pos = enuToRealLatLng(entity.position, globeConfig.originLat, globeConfig.originLng);
      bounds.extend({ lat: pos.lat, lng: pos.lng });
    }

    if (state.batteries) {
      for (const bat of state.batteries) {
        const pos = enuToRealLatLng(bat.position, globeConfig.originLat, globeConfig.originLng);
        bounds.extend({ lat: pos.lat, lng: pos.lng });
      }
    }

    if (state.protected_areas) {
      for (const area of state.protected_areas) {
        const pos = enuToRealLatLng(area.center, globeConfig.originLat, globeConfig.originLng);
        bounds.extend({ lat: pos.lat, lng: pos.lng });
      }
    }

    map.fitBounds(bounds, { top: 60, right: 60, bottom: 120, left: 60 });
  }, [map, state?.run_id, state?.entities, state?.batteries, state?.protected_areas, globeConfig]);

  // Auto-fit to preview batteries/areas when no simulation is running
  useEffect(() => {
    if (!map || state) return; // only when sim is not running
    if (hasFittedPreviewRef.current) return;

    const hasPreviews = (previewBatteries && previewBatteries.length > 0) ||
                        (previewProtectedAreas && previewProtectedAreas.length > 0);
    if (!hasPreviews) return;

    const bounds = new google.maps.LatLngBounds();

    if (previewBatteries) {
      for (const bat of previewBatteries) {
        const pos = enuToRealLatLng(bat.position, globeConfig.originLat, globeConfig.originLng);
        bounds.extend({ lat: pos.lat, lng: pos.lng });
      }
    }

    if (previewProtectedAreas) {
      for (const area of previewProtectedAreas) {
        const pos = enuToRealLatLng(area.center, globeConfig.originLat, globeConfig.originLng);
        bounds.extend({ lat: pos.lat, lng: pos.lng });
      }
    }

    map.fitBounds(bounds, { top: 60, right: 60, bottom: 120, left: 60 });
    hasFittedPreviewRef.current = true;
  }, [map, state, previewBatteries, previewProtectedAreas, globeConfig]);

  // Reset preview fit flag when preview data changes
  useEffect(() => {
    hasFittedPreviewRef.current = false;
  }, [previewBatteries, previewProtectedAreas]);

  return null;
}
