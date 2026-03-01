/**
 * KeyboardShortcutsPanel — Floating overlay listing all shortcuts
 *
 * Toggle with `?` key.
 */

import type { ShortcutDef } from '../hooks/useKeyboardShortcuts';

interface KeyboardShortcutsPanelProps {
  shortcuts: ShortcutDef[];
  onClose: () => void;
}

export function KeyboardShortcutsPanel({ shortcuts, onClose }: KeyboardShortcutsPanelProps) {
  return (
    <div className="shortcuts-backdrop" onClick={onClose}>
      <div className="shortcuts-panel" onClick={(e) => e.stopPropagation()}>
        <div className="shortcuts-header">
          <span className="shortcuts-title">KEYBOARD SHORTCUTS</span>
          <button className="shortcuts-close" onClick={onClose}>&times;</button>
        </div>
        <div className="shortcuts-grid">
          {shortcuts.map((s) => (
            <div key={s.key} className="shortcut-row">
              <kbd className="shortcut-key">{s.label}</kbd>
              <span className="shortcut-desc">{s.description}</span>
            </div>
          ))}
        </div>
        <div className="shortcuts-hint">Press <kbd>?</kbd> to toggle this panel</div>
      </div>
    </div>
  );
}
