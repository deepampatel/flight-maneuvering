/**
 * Dark mode fallback styles for Google Maps.
 *
 * The primary approach uses `colorScheme="DARK"` on the Map component (native
 * Google Maps v3.56+). These JSON styles serve as a deeper customization for
 * a military-C2 aesthetic if needed.
 */

export const DARK_MAP_STYLES: google.maps.MapTypeStyle[] = [
  { elementType: 'geometry', stylers: [{ color: '#0d1117' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#6e7681' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#0d1117' }] },
  { featureType: 'administrative', elementType: 'geometry', stylers: [{ color: '#21262d' }] },
  { featureType: 'administrative.country', elementType: 'labels.text.fill', stylers: [{ color: '#8b949e' }] },
  { featureType: 'landscape', elementType: 'geometry', stylers: [{ color: '#0d1117' }] },
  { featureType: 'poi', stylers: [{ visibility: 'off' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#161b22' }] },
  { featureType: 'road', elementType: 'labels', stylers: [{ visibility: 'off' }] },
  { featureType: 'transit', stylers: [{ visibility: 'off' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#010409' }] },
  { featureType: 'water', elementType: 'labels', stylers: [{ visibility: 'off' }] },
];

/** Tier colors matching the 3D scene design system. */
export const TIER_COLORS: Record<string, string> = {
  iron_dome: '#3b82f6',
  davids_sling: '#06b6d4',
  arrow: '#8b5cf6',
};

/** Entity type colors matching the 3D scene. */
export const ENTITY_COLORS = {
  target: '#ef4444',
  interceptor: '#3b82f6',
  battery: '#eab308',
  launcher: '#22c55e',
} as const;
