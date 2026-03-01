/**
 * ScenarioBuilder — Slide-out left drawer for configuring scenarios
 *
 * Accordion sections: Presets, Defense Layers, Threat Waves,
 * Protected Areas, Engagement, Environment.
 */

import { useState, useCallback } from 'react';
import { ScenarioPresets } from './builder/ScenarioPresets';
import { DefenseLayerEditor } from './builder/DefenseLayerEditor';
import { ThreatWaveEditor } from './builder/ThreatWaveEditor';
import { ProtectedAreaEditor } from './builder/ProtectedAreaEditor';
import { EngagementSettings } from './builder/EngagementSettings';
import { EnvironmentSettings } from './builder/EnvironmentSettings';
import type { NarrativeScenario } from '../data/scenarios';
import type {
  ScenarioBuilderState,
  BuilderBatteryConfig,
  BuilderWaveConfig,
  BuilderProtectedArea,
} from '../types';

interface ScenarioBuilderProps {
  open: boolean;
  onToggle: () => void;
  isRunning: boolean;
  connected: boolean;
  isRecording: boolean;
  onLaunch: (state: ScenarioBuilderState, presetScenario: string | null) => void;
  onStop: () => void;
  onToggleRecording: () => void;
}

const DEFAULT_STATE: ScenarioBuilderState = {
  batteries: [],
  waves: [],
  protectedAreas: [],
  guidance: 'proportional_nav',
  navConstant: 4,
  evasion: 'none',
  wtaAlgorithm: 'hungarian',
  killRadius: 150,
  numInterceptors: 1,
  numTargets: 1,
  windSpeed: 0,
  windDirection: 0,
  windGusts: 0,
  enableDrag: false,
  enableTerrain: false,
  enableDatalink: false,
  enableCooperative: false,
  enableSwarm: false,
  enableHmt: false,
  hmtAuthorityLevel: 'full_auto',
  maxTime: 60,
};

type SectionId = 'presets' | 'defense' | 'waves' | 'areas' | 'engagement' | 'environment';

export function ScenarioBuilder({
  open, onToggle, isRunning, connected,
  isRecording, onLaunch, onStop, onToggleRecording,
}: ScenarioBuilderProps) {
  const [state, setState] = useState<ScenarioBuilderState>(DEFAULT_STATE);
  const [activePreset, setActivePreset] = useState<string | null>('first-contact');
  const [presetScenarioId, setPresetScenarioId] = useState<string | null>('head_on');
  const [expandedSections, setExpandedSections] = useState<Set<SectionId>>(
    new Set(['presets', 'defense', 'waves'])
  );

  const toggleSection = (id: SectionId) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handlePresetSelect = useCallback((scenario: NarrativeScenario | null) => {
    if (!scenario) {
      setActivePreset(null);
      setPresetScenarioId(null);
      setState(DEFAULT_STATE);
      return;
    }
    setActivePreset(scenario.id);
    setPresetScenarioId(scenario.config.scenario);
    setState({
      ...DEFAULT_STATE,
      guidance: scenario.config.guidance,
      navConstant: scenario.config.navConstant,
      evasion: scenario.config.evasion,
      numInterceptors: scenario.config.numInterceptors,
      numTargets: scenario.config.numTargets,
      enableCooperative: scenario.config.enableCooperative || false,
      enableSwarm: scenario.config.enableSwarm || false,
      enableHmt: scenario.config.enableHMT || false,
      enableDrag: scenario.config.enableDrag || false,
      windSpeed: scenario.config.windSpeed || 0,
      windDirection: scenario.config.windDirection || 0,
    });
  }, []);

  const updateField = useCallback((field: string, value: string | number | boolean) => {
    setState((prev) => ({ ...prev, [field]: value }));
    // When any setting changes, clear preset since we're customizing
    if (activePreset !== null) {
      setActivePreset(null);
      setPresetScenarioId(null);
    }
  }, [activePreset]);

  const handleLaunch = () => {
    onLaunch(state, presetScenarioId);
  };

  return (
    <>
      {/* Toggle tab on the edge */}
      <button className={`builder-toggle ${open ? 'open' : ''}`} onClick={onToggle}>
        {open ? '◂' : '▸'}
      </button>

      <div className={`scenario-builder ${open ? 'open' : ''} ${isRunning ? 'running' : ''}`}>
        <div className="builder-header">
          <h2>Scenario Builder</h2>
        </div>

        <div className="builder-scroll">
          {/* Presets */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('presets')}>
              <span className="section-chevron">{expandedSections.has('presets') ? '▾' : '▸'}</span>
              Quick Start
              {activePreset && <span className="section-badge">Preset</span>}
            </button>
            {expandedSections.has('presets') && (
              <div className="section-content">
                <ScenarioPresets activeId={activePreset} onSelect={handlePresetSelect} disabled={isRunning} />
              </div>
            )}
          </div>

          {/* Defense Layers */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('defense')}>
              <span className="section-chevron">{expandedSections.has('defense') ? '▾' : '▸'}</span>
              Defense Layers
              {state.batteries.length > 0 && (
                <span className="section-count">{state.batteries.length}</span>
              )}
            </button>
            {expandedSections.has('defense') && (
              <div className="section-content">
                <DefenseLayerEditor
                  batteries={state.batteries}
                  onChange={(batteries: BuilderBatteryConfig[]) => setState((s) => ({ ...s, batteries }))}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>

          {/* Threat Waves */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('waves')}>
              <span className="section-chevron">{expandedSections.has('waves') ? '▾' : '▸'}</span>
              Threat Waves
              {state.waves.length > 0 && (
                <span className="section-count">{state.waves.length}</span>
              )}
            </button>
            {expandedSections.has('waves') && (
              <div className="section-content">
                <ThreatWaveEditor
                  waves={state.waves}
                  onChange={(waves: BuilderWaveConfig[]) => setState((s) => ({ ...s, waves }))}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>

          {/* Protected Areas */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('areas')}>
              <span className="section-chevron">{expandedSections.has('areas') ? '▾' : '▸'}</span>
              Protected Areas
              {state.protectedAreas.length > 0 && (
                <span className="section-count">{state.protectedAreas.length}</span>
              )}
            </button>
            {expandedSections.has('areas') && (
              <div className="section-content">
                <ProtectedAreaEditor
                  areas={state.protectedAreas}
                  onChange={(protectedAreas: BuilderProtectedArea[]) => setState((s) => ({ ...s, protectedAreas }))}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>

          {/* Engagement */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('engagement')}>
              <span className="section-chevron">{expandedSections.has('engagement') ? '▾' : '▸'}</span>
              Engagement
            </button>
            {expandedSections.has('engagement') && (
              <div className="section-content">
                <EngagementSettings
                  guidance={state.guidance}
                  navConstant={state.navConstant}
                  evasion={state.evasion}
                  wtaAlgorithm={state.wtaAlgorithm}
                  killRadius={state.killRadius}
                  numInterceptors={state.numInterceptors}
                  numTargets={state.numTargets}
                  onChange={updateField}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>

          {/* Environment */}
          <div className="builder-section">
            <button className="section-toggle" onClick={() => toggleSection('environment')}>
              <span className="section-chevron">{expandedSections.has('environment') ? '▾' : '▸'}</span>
              Environment
            </button>
            {expandedSections.has('environment') && (
              <div className="section-content">
                <EnvironmentSettings
                  windSpeed={state.windSpeed}
                  windDirection={state.windDirection}
                  windGusts={state.windGusts}
                  enableDrag={state.enableDrag}
                  enableTerrain={state.enableTerrain}
                  enableDatalink={state.enableDatalink}
                  enableCooperative={state.enableCooperative}
                  enableSwarm={state.enableSwarm}
                  enableHmt={state.enableHmt}
                  hmtAuthorityLevel={state.hmtAuthorityLevel}
                  maxTime={state.maxTime}
                  onChange={updateField}
                  disabled={isRunning}
                />
              </div>
            )}
          </div>
        </div>

        {/* Action bar */}
        <div className="builder-actions">
          {!isRunning ? (
            <button className="builder-launch-btn" onClick={handleLaunch} disabled={!connected}>
              Launch
            </button>
          ) : (
            <button className="builder-stop-btn" onClick={onStop}>
              Stop
            </button>
          )}
          <button
            className={`builder-rec-btn ${isRecording ? 'recording' : ''}`}
            onClick={onToggleRecording}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            REC
          </button>
        </div>
      </div>
    </>
  );
}
