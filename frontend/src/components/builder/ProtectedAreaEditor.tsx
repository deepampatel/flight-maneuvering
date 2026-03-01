/**
 * ProtectedAreaEditor — List of protected zones with add/remove
 */

import type { BuilderProtectedArea } from '../../types';

interface ProtectedAreaEditorProps {
  areas: BuilderProtectedArea[];
  onChange: (areas: BuilderProtectedArea[]) => void;
  disabled: boolean;
}

let areaCounter = 0;

export function createDefaultArea(): BuilderProtectedArea {
  areaCounter++;
  const names = ['Alpha City', 'Bravo City', 'Charlie City', 'Delta City', 'Echo City', 'Foxtrot City'];
  return {
    id: `area_${Date.now()}_${areaCounter}`,
    name: names[(areaCounter - 1) % names.length],
    center: { x: 0, y: 0, z: 0 },
    radius: 2000,
    priority: 5,
  };
}

export function ProtectedAreaEditor({ areas, onChange, disabled }: ProtectedAreaEditorProps) {
  const addArea = () => {
    onChange([...areas, createDefaultArea()]);
  };

  const updateArea = (index: number, field: string, value: string | number) => {
    const next = [...areas];
    const area = { ...next[index] };
    if (field === 'name') area.name = value as string;
    else if (field === 'radius') area.radius = +value;
    else if (field === 'priority') area.priority = +value;
    else if (field === 'cx') area.center = { ...area.center, x: +value };
    else if (field === 'cy') area.center = { ...area.center, y: +value };
    next[index] = area;
    onChange(next);
  };

  const removeArea = (index: number) => {
    onChange(areas.filter((_, i) => i !== index));
  };

  return (
    <div className="protected-area-editor">
      {areas.length === 0 && (
        <div className="builder-empty">No protected areas</div>
      )}
      {areas.map((area, i) => (
        <div key={area.id} className="area-card">
          <div className="area-card-header">
            <input
              className="area-name-input"
              value={area.name}
              onChange={(e) => updateArea(i, 'name', e.target.value)}
              disabled={disabled}
            />
            <span className="area-priority">P{area.priority}</span>
            <button className="area-remove" onClick={() => removeArea(i)} disabled={disabled}>&times;</button>
          </div>
          <div className="area-fields">
            <div className="area-field">
              <label>X</label>
              <input type="number" value={area.center.x} disabled={disabled}
                onChange={(e) => updateArea(i, 'cx', e.target.value)} />
            </div>
            <div className="area-field">
              <label>Y</label>
              <input type="number" value={area.center.y} disabled={disabled}
                onChange={(e) => updateArea(i, 'cy', e.target.value)} />
            </div>
            <div className="area-field">
              <label>Radius</label>
              <input type="number" min={500} max={10000} step={100} value={area.radius} disabled={disabled}
                onChange={(e) => updateArea(i, 'radius', e.target.value)} />
            </div>
            <div className="area-field">
              <label>Priority</label>
              <input type="range" min={1} max={10} value={area.priority} disabled={disabled}
                onChange={(e) => updateArea(i, 'priority', e.target.value)} />
            </div>
          </div>
        </div>
      ))}
      <button className="builder-add-btn" onClick={addArea} disabled={disabled}>
        + Add Zone
      </button>
    </div>
  );
}
