/**
 * ViewToggle — SIM / Globe / Map view mode switcher
 *
 * Three-button toggle matching the camera mode selector styling.
 * Positioned above the camera mode buttons in the bottom-right.
 */

import type { ViewMode } from './CameraController';

interface ViewToggleProps {
  viewMode: ViewMode;
  onSetViewMode: (mode: ViewMode) => void;
}

export function ViewToggle({ viewMode, onSetViewMode }: ViewToggleProps) {
  return (
    <div className="view-toggle">
      <button
        className={`view-toggle-btn ${viewMode === 'sim' ? 'active' : ''}`}
        onClick={() => onSetViewMode('sim')}
        title="Simulation 3D View (M)"
      >
        SIM
      </button>
      <button
        className={`view-toggle-btn ${viewMode === 'globe' ? 'active' : ''}`}
        onClick={() => onSetViewMode('globe')}
        title="3D Globe View (M)"
      >
        GLOBE
      </button>
      <button
        className={`view-toggle-btn ${viewMode === 'map' ? 'active' : ''}`}
        onClick={() => onSetViewMode('map')}
        title="2D Map View (M)"
      >
        MAP
      </button>
    </div>
  );
}
