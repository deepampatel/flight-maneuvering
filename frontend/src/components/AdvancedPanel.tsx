/**
 * AdvancedPanel — Tabbed sidebar for analysis and configuration
 *
 * Right-side slide-out panel with tabs: Analysis, Recordings,
 * Environment, ML/AI, Swarm, HMT.
 */

import { useState } from 'react';
import { AnalysisTab } from './panels/AnalysisTab';
import { RecordingsTab } from './panels/RecordingsTab';
import { EnvironmentTab } from './panels/EnvironmentTab';
import { MLTab } from './panels/MLTab';
import { SwarmTab } from './panels/SwarmTab';
import { HMTTab } from './panels/HMTTab';
import type {
  MonteCarloResults,
  EnvelopeResults,
  RecordingMetadata,
  EnvironmentState,
  CooperativeState,
  EngagementZoneCreateRequest,
  MLStatus,
  SwarmStatus,
  SwarmConfig,
  FormationInfo,
  FormationType,
  HMTStatus,
  HMTConfig,
  PendingAction,
  AuthorityLevel,
  AuthorityLevelInfo,
} from '../types';

type TabId = 'analysis' | 'recordings' | 'environment' | 'ml' | 'swarm' | 'hmt';

interface AdvancedPanelProps {
  isRunning: boolean;
  // Analysis
  selectedScenario: string;
  selectedGuidance: string;
  navConstant: number;
  selectedEvasion: string;
  onRunMonteCarlo: (options: {
    scenario: string;
    guidance: string;
    navConstant: number;
    numRuns: number;
    killRadius: number;
    positionNoiseStd: number;
    velocityNoiseStd: number;
  }) => Promise<MonteCarloResults>;
  onRunEnvelope: (config: {
    guidance: string;
    nav_constant: number;
    evasion: string;
    range_steps: number;
    bearing_steps: number;
    runs_per_point: number;
  }) => Promise<EnvelopeResults>;
  monteCarloLoading: boolean;
  envelopeLoading: boolean;
  // Recordings
  recordings: RecordingMetadata[];
  onStartReplay: (id: string) => void;
  onDeleteRecording: (id: string) => void;
  // Environment
  windSpeed: number;
  windDirection: number;
  windGusts: number;
  enableDrag: boolean;
  onWindSpeed: (v: number) => void;
  onWindDirection: (v: number) => void;
  onWindGusts: (v: number) => void;
  onEnableDrag: (v: boolean) => void;
  enableCooperative: boolean;
  onEnableCooperative: (v: boolean) => void;
  environmentState: EnvironmentState | null;
  cooperativeState?: CooperativeState | null;
  onCreateEngagementZone?: (zone: EngagementZoneCreateRequest) => void;
  // ML
  mlStatus?: MLStatus | null;
  onFetchMLStatus?: () => void;
  // Swarm
  enableSwarm: boolean;
  swarmFormation: FormationType;
  swarmSpacing: number;
  onEnableSwarm: (v: boolean) => void;
  onSwarmFormation: (v: FormationType) => void;
  onSwarmSpacing: (v: number) => void;
  swarmStatus?: SwarmStatus | null;
  formationTypes?: FormationInfo[];
  onFetchSwarmStatus?: () => void;
  onConfigureSwarm?: (config: Partial<SwarmConfig>) => void;
  onSetSwarmFormation?: (formation: FormationType) => void;
  // HMT
  enableHmt: boolean;
  hmtAuthorityLevel: AuthorityLevel;
  onEnableHmt: (v: boolean) => void;
  onHmtAuthorityLevel: (v: AuthorityLevel) => void;
  hmtStatus?: HMTStatus | null;
  authorityLevels?: AuthorityLevelInfo[];
  pendingActions?: PendingAction[];
  onFetchHMTStatus?: () => void;
  onFetchPendingActions?: () => void;
  onApproveAction?: (actionId: string, reason?: string) => void;
  onRejectAction?: (actionId: string, reason?: string) => void;
  onSetAuthorityLevel?: (level: AuthorityLevel) => void;
  onConfigureHMT?: (config: Partial<HMTConfig>) => void;
}

const TABS: { id: TabId; label: string }[] = [
  { id: 'analysis', label: 'Analysis' },
  { id: 'recordings', label: 'Recordings' },
  { id: 'environment', label: 'Environment' },
  { id: 'ml', label: 'ML/AI' },
  { id: 'swarm', label: 'Swarm' },
  { id: 'hmt', label: 'HMT' },
];

export function AdvancedPanel(props: AdvancedPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId | null>(null);
  const [mcResults, setMcResults] = useState<MonteCarloResults | null>(null);
  const [envelopeResults, setEnvelopeResults] = useState<EnvelopeResults | null>(null);

  const toggleTab = (tab: TabId) => {
    if (activeTab === tab) {
      setActiveTab(null);
    } else {
      setActiveTab(tab);
      // Trigger fetches for tabs that need fresh data
      if (tab === 'ml' && props.onFetchMLStatus) props.onFetchMLStatus();
      if (tab === 'swarm' && props.onFetchSwarmStatus) props.onFetchSwarmStatus();
      if (tab === 'hmt') {
        if (props.onFetchHMTStatus) props.onFetchHMTStatus();
        if (props.onFetchPendingActions) props.onFetchPendingActions();
      }
    }
  };

  const recordingsLabel = `Recordings${props.recordings.length > 0 ? ` (${props.recordings.length})` : ''}`;
  const hmtLabel = `HMT${props.pendingActions && props.pendingActions.length > 0 ? ` (${props.pendingActions.length})` : ''}`;

  return (
    <div className="advanced-panel">
      <div className="advanced-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => toggleTab(tab.id)}
          >
            {tab.id === 'recordings' ? recordingsLabel : tab.id === 'hmt' ? hmtLabel : tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'analysis' && (
        <AnalysisTab
          selectedScenario={props.selectedScenario}
          selectedGuidance={props.selectedGuidance}
          navConstant={props.navConstant}
          selectedEvasion={props.selectedEvasion}
          onRunMonteCarlo={props.onRunMonteCarlo}
          onRunEnvelope={props.onRunEnvelope}
          monteCarloLoading={props.monteCarloLoading}
          envelopeLoading={props.envelopeLoading}
          mcResults={mcResults}
          envelopeResults={envelopeResults}
          onMcResults={setMcResults}
          onEnvelopeResults={setEnvelopeResults}
        />
      )}

      {activeTab === 'recordings' && (
        <RecordingsTab
          recordings={props.recordings}
          onStartReplay={props.onStartReplay}
          onDeleteRecording={props.onDeleteRecording}
        />
      )}

      {activeTab === 'environment' && (
        <EnvironmentTab
          isRunning={props.isRunning}
          windSpeed={props.windSpeed}
          windDirection={props.windDirection}
          windGusts={props.windGusts}
          enableDrag={props.enableDrag}
          onWindSpeed={props.onWindSpeed}
          onWindDirection={props.onWindDirection}
          onWindGusts={props.onWindGusts}
          onEnableDrag={props.onEnableDrag}
          enableCooperative={props.enableCooperative}
          onEnableCooperative={props.onEnableCooperative}
          environmentState={props.environmentState}
          cooperativeState={props.cooperativeState}
          onCreateEngagementZone={props.onCreateEngagementZone}
        />
      )}

      {activeTab === 'ml' && (
        <MLTab mlStatus={props.mlStatus} />
      )}

      {activeTab === 'swarm' && (
        <SwarmTab
          isRunning={props.isRunning}
          enableSwarm={props.enableSwarm}
          swarmFormation={props.swarmFormation}
          swarmSpacing={props.swarmSpacing}
          onEnableSwarm={props.onEnableSwarm}
          onSwarmFormation={props.onSwarmFormation}
          onSwarmSpacing={props.onSwarmSpacing}
          swarmStatus={props.swarmStatus}
          formationTypes={props.formationTypes}
          onConfigureSwarm={props.onConfigureSwarm}
          onSetSwarmFormation={props.onSetSwarmFormation}
        />
      )}

      {activeTab === 'hmt' && (
        <HMTTab
          isRunning={props.isRunning}
          enableHmt={props.enableHmt}
          hmtAuthorityLevel={props.hmtAuthorityLevel}
          onEnableHmt={props.onEnableHmt}
          onHmtAuthorityLevel={props.onHmtAuthorityLevel}
          hmtStatus={props.hmtStatus}
          authorityLevels={props.authorityLevels}
          pendingActions={props.pendingActions}
          onSetAuthorityLevel={props.onSetAuthorityLevel}
          onConfigureHMT={props.onConfigureHMT}
          onApproveAction={props.onApproveAction}
          onRejectAction={props.onRejectAction}
        />
      )}
    </div>
  );
}
