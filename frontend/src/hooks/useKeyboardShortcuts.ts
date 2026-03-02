/**
 * useKeyboardShortcuts — Global keyboard shortcut handler
 *
 * Maps keyboard events to simulation actions. Only active when
 * no input/textarea/modal is focused.
 */

import { useEffect, useCallback, useState } from 'react';

export interface ShortcutDef {
  key: string;
  label: string;
  description: string;
}

export const SHORTCUTS: ShortcutDef[] = [
  { key: 'Space', label: 'SPACE', description: 'Launch / Pause simulation' },
  { key: 'Escape', label: 'ESC', description: 'Stop / Abort simulation' },
  { key: 'r', label: 'R', description: 'Toggle recording' },
  { key: '1', label: '1', description: 'Free camera' },
  { key: '2', label: '2', description: 'Chase camera' },
  { key: '3', label: '3', description: 'Tactical (top-down) camera' },
  { key: '4', label: '4', description: 'Cinematic camera' },
  { key: 'm', label: 'M', description: 'Cycle view: SIM → Globe → Map' },
  { key: 'f', label: 'F', description: 'Focus selected entity (or all)' },
  { key: 'a', label: 'A', description: 'Toggle advanced panel' },
  { key: '?', label: '?', description: 'Show keyboard shortcuts' },
];

interface ShortcutActions {
  onLaunch?: () => void;
  onStop?: () => void;
  onToggleRecording?: () => void;
  onCameraMode?: (mode: string) => void;
  onToggleAdvanced?: () => void;
  onToggleViewMode?: () => void;
  onFocusEntity?: () => void;
  isRunning?: boolean;
}

export function useKeyboardShortcuts(actions: ShortcutActions) {
  const [showHelp, setShowHelp] = useState(false);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Skip when typing in inputs
    const tag = (e.target as HTMLElement).tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    switch (e.key) {
      case ' ':
        e.preventDefault();
        if (actions.isRunning) {
          actions.onStop?.();
        } else {
          actions.onLaunch?.();
        }
        break;
      case 'Escape':
        if (showHelp) {
          setShowHelp(false);
        } else {
          actions.onStop?.();
        }
        break;
      case 'r':
      case 'R':
        actions.onToggleRecording?.();
        break;
      case '1':
        actions.onCameraMode?.('free');
        break;
      case '2':
        actions.onCameraMode?.('chase');
        break;
      case '3':
        actions.onCameraMode?.('tactical');
        break;
      case '4':
        actions.onCameraMode?.('cinematic');
        break;
      case 'm':
      case 'M':
        actions.onToggleViewMode?.();
        break;
      case 'f':
      case 'F':
        actions.onFocusEntity?.();
        break;
      case 'a':
      case 'A':
        actions.onToggleAdvanced?.();
        break;
      case '?':
        setShowHelp(prev => !prev);
        break;
    }
  }, [actions, showHelp]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return { showHelp, setShowHelp, shortcuts: SHORTCUTS };
}
