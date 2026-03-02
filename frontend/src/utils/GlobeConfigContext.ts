/**
 * Scene configuration context — provides viewMode + globe config to all 3D components.
 *
 * Backward-compatible: useGlobe() still returns GlobeConfig.
 * New code should use useSceneConfig() to get viewMode + globeConfig.
 */

import { createContext, useContext } from 'react';
import { DEFAULT_GLOBE_CONFIG } from './globeCoords';
import type { GlobeConfig } from './globeCoords';

export interface SceneConfig {
  viewMode: 'sim' | 'globe';
  globeConfig: GlobeConfig;
}

const defaultSceneConfig: SceneConfig = {
  viewMode: 'globe',
  globeConfig: DEFAULT_GLOBE_CONFIG,
};

export const SceneConfigContext = createContext<SceneConfig>(defaultSceneConfig);

/** Get full scene config (viewMode + globeConfig). */
export function useSceneConfig(): SceneConfig {
  return useContext(SceneConfigContext);
}

/** Backward-compatible: get just the GlobeConfig. */
export function useGlobe(): GlobeConfig {
  return useContext(SceneConfigContext).globeConfig;
}

// Alias for backward compatibility with existing imports
export const GlobeConfigContext = SceneConfigContext;
