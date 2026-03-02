/**
 * MapOverlays — imperative Google Maps shapes (trails, range circles, areas).
 *
 * These use the raw google.maps API via useMap() because @vis.gl/react-google-maps
 * doesn't have React components for Polyline, Circle, or Polygon.
 *
 * Shapes are managed imperatively: create on mount, update on state change, cleanup on unmount.
 */

import { useEffect, useRef } from 'react';
import { useMap } from '@vis.gl/react-google-maps';
import type { SimStateEvent, SimStateEventWithEnvironment, InterceptGeometry, BuilderProtectedArea } from '../../types';
import type { GlobeConfig } from '../../utils/globeCoords';
import { enuToRealLatLng } from '../../utils/globeCoords';
import { TIER_COLORS, ENTITY_COLORS } from './mapStyles';

const MAX_TRAIL_LENGTH = 500;

interface MapOverlaysProps {
  state: SimStateEvent | SimStateEventWithEnvironment | null;
  globeConfig: GlobeConfig;
  interceptGeometry?: InterceptGeometry[] | null;
  previewProtectedAreas?: BuilderProtectedArea[];
}

export function MapOverlays({
  state,
  globeConfig,
  interceptGeometry,
  previewProtectedAreas,
}: MapOverlaysProps) {
  const map = useMap();

  // Refs for imperative Google Maps objects
  const trailsRef = useRef<Map<string, { polyline: google.maps.Polyline; path: google.maps.LatLng[] }>>(new Map());
  const circlesRef = useRef<google.maps.Circle[]>([]);
  const interceptLinesRef = useRef<google.maps.Polyline[]>([]);
  const prevRunIdRef = useRef<string | null>(null);

  // =========================================================================
  // Entity trails — accumulate positions across ticks
  // =========================================================================
  useEffect(() => {
    if (!map || !state?.entities) return;

    // Clear trails on new run
    if (state.run_id !== prevRunIdRef.current) {
      prevRunIdRef.current = state.run_id;
      trailsRef.current.forEach(({ polyline }) => polyline.setMap(null));
      trailsRef.current.clear();
    }

    const interceptedIds = new Set<string>();
    if (state.intercepted_pairs) {
      for (const pair of state.intercepted_pairs) {
        interceptedIds.add(pair[0]);
        interceptedIds.add(pair[1]);
      }
    }

    const activeIds = new Set<string>();

    for (const entity of state.entities) {
      if (interceptedIds.has(entity.id)) continue;
      activeIds.add(entity.id);

      const pos = enuToRealLatLng(entity.position, globeConfig.originLat, globeConfig.originLng);
      const latLng = new google.maps.LatLng(pos.lat, pos.lng);

      let trail = trailsRef.current.get(entity.id);

      if (!trail) {
        // Create new trail
        const isTarget = entity.type === 'target';
        const color = isTarget ? ENTITY_COLORS.target : ENTITY_COLORS.interceptor;
        const polyline = new google.maps.Polyline({
          map,
          strokeColor: color,
          strokeOpacity: 0.6,
          strokeWeight: 2,
          geodesic: true,
        });
        trail = { polyline, path: [] };
        trailsRef.current.set(entity.id, trail);
      }

      // Append position (throttled by minimum distance)
      const lastPos = trail.path[trail.path.length - 1];
      if (!lastPos || google.maps.geometry?.spherical?.computeDistanceBetween(lastPos, latLng) > 10) {
        trail.path.push(latLng);
        // Trim if too long
        if (trail.path.length > MAX_TRAIL_LENGTH) {
          trail.path = trail.path.slice(Math.floor(MAX_TRAIL_LENGTH * 0.1));
        }
        trail.polyline.setPath(trail.path);
      }
    }

    // Don't remove trails of dead entities — keep as persistent traces
  }, [map, state?.tick, state?.entities, state?.intercepted_pairs, state?.run_id, globeConfig]);

  // =========================================================================
  // Static circles — battery ranges, protected areas
  // =========================================================================
  useEffect(() => {
    if (!map) return;

    // Clear previous circles
    circlesRef.current.forEach(c => c.setMap(null));
    circlesRef.current = [];

    // Battery radar ranges
    if (state?.batteries) {
      for (const battery of state.batteries) {
        const pos = enuToRealLatLng(battery.position, globeConfig.originLat, globeConfig.originLng);
        const tierColor = TIER_COLORS[battery.tier] || ENTITY_COLORS.battery;
        const circle = new google.maps.Circle({
          map,
          center: { lat: pos.lat, lng: pos.lng },
          radius: battery.radar_range, // meters — Google Maps Circle accepts meters
          fillColor: tierColor,
          fillOpacity: 0.04,
          strokeColor: tierColor,
          strokeOpacity: 0.3,
          strokeWeight: 1,
          clickable: false,
        });
        circlesRef.current.push(circle);
      }
    }

    // Protected areas (live)
    if (state?.protected_areas) {
      for (const area of state.protected_areas) {
        const pos = enuToRealLatLng(area.center, globeConfig.originLat, globeConfig.originLng);
        const circle = new google.maps.Circle({
          map,
          center: { lat: pos.lat, lng: pos.lng },
          radius: area.radius, // meters
          fillColor: '#22c55e',
          fillOpacity: 0.06,
          strokeColor: '#22c55e',
          strokeOpacity: 0.4,
          strokeWeight: 1.5,
          clickable: false,
        });
        circlesRef.current.push(circle);
      }
    }

    // Preview protected areas (before simulation)
    if (!state && previewProtectedAreas) {
      for (const area of previewProtectedAreas) {
        const pos = enuToRealLatLng(area.center, globeConfig.originLat, globeConfig.originLng);
        const circle = new google.maps.Circle({
          map,
          center: { lat: pos.lat, lng: pos.lng },
          radius: area.radius,
          fillColor: '#22c55e',
          fillOpacity: 0.06,
          strokeColor: '#22c55e',
          strokeOpacity: 0.3,
          strokeWeight: 1,
          clickable: false,
        });
        circlesRef.current.push(circle);
      }
    }

    return () => {
      circlesRef.current.forEach(c => c.setMap(null));
      circlesRef.current = [];
    };
  }, [map, state?.batteries, state?.protected_areas, previewProtectedAreas, globeConfig, state]);

  // =========================================================================
  // Intercept geometry lines
  // =========================================================================
  useEffect(() => {
    if (!map) return;

    // Clear previous
    interceptLinesRef.current.forEach(l => l.setMap(null));
    interceptLinesRef.current = [];

    if (!interceptGeometry || !state?.entities) return;

    const interceptedIds = new Set<string>();
    if (state.intercepted_pairs) {
      for (const pair of state.intercepted_pairs) {
        interceptedIds.add(pair[0]);
        interceptedIds.add(pair[1]);
      }
    }

    for (const geom of interceptGeometry) {
      if (!geom.intercept_point) continue;
      if (interceptedIds.has(geom.interceptor_id)) continue;
      if (interceptedIds.has(geom.target_id)) continue;

      const interceptor = state.entities.find(e => e.id === geom.interceptor_id);
      if (!interceptor) continue;

      const from = enuToRealLatLng(interceptor.position, globeConfig.originLat, globeConfig.originLng);
      const to = enuToRealLatLng(geom.intercept_point, globeConfig.originLat, globeConfig.originLng);

      const line = new google.maps.Polyline({
        map,
        path: [
          { lat: from.lat, lng: from.lng },
          { lat: to.lat, lng: to.lng },
        ],
        strokeColor: geom.collision_course ? '#22c55e' : '#f97316',
        strokeOpacity: 0.5,
        strokeWeight: 1,
        geodesic: true,
        icons: [{
          icon: { path: 'M 0,-1 0,1', strokeOpacity: 0.5, scale: 3 },
          offset: '0',
          repeat: '12px',
        }],
      });
      interceptLinesRef.current.push(line);
    }

    return () => {
      interceptLinesRef.current.forEach(l => l.setMap(null));
      interceptLinesRef.current = [];
    };
  }, [map, interceptGeometry, state?.entities, state?.intercepted_pairs, globeConfig]);

  // =========================================================================
  // Cleanup on unmount
  // =========================================================================
  useEffect(() => {
    return () => {
      trailsRef.current.forEach(({ polyline }) => polyline.setMap(null));
      trailsRef.current.clear();
      circlesRef.current.forEach(c => c.setMap(null));
      circlesRef.current = [];
      interceptLinesRef.current.forEach(l => l.setMap(null));
      interceptLinesRef.current = [];
    };
  }, []);

  return null;
}
