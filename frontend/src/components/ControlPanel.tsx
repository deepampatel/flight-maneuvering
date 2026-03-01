/**
 * ControlPanel — Thin orchestration wrapper
 *
 * Composes MissionToolbar (top bar), MissionStatusHUD (bottom
 * overlay), and AdvancedPanel (slide-out sidebar). Owns form state
 * and data-fetching side-effects.
 *
 * Decomposed from the original 1,315-line monolith into focused
 * sub-components under components/panels/.
 */

import { useState, useEffect } from 'react';
import { MissionToolbar } from './MissionToolbar';
import { MissionStatusHUD } from './MissionStatusHUD';
import { AdvancedPanel } from './AdvancedPanel';
import type {
  SimStateEvent,
  Scenario,
  GuidanceLaw,
  EvasionType,
  MonteCarloResults,
  EnvelopeResults,
  InterceptGeometry,
  ThreatAssessment,
  RecordingMetadata,
  ReplayState,
  WTAAlgorithm,
  AssignmentResult,
  EnvironmentState,
  CooperativeState,
  EngagementZoneCreateRequest,
  HandoffRequestCreate,
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

interface ControlPanelProps {
  connected: boolean;
  state: SimStateEvent | null;
  scenarios: Record<string, Scenario>;
  guidanceLaws: GuidanceLaw[];
  evasionTypes: EvasionType[];
  onStart: (options: {
    scenario: string;
    guidance: string;
    navConstant: number;
    evasion: string;
    numInterceptors: number;
    numTargets?: number;
    wtaAlgorithm?: string;
    windSpeed?: number;
    windDirection?: number;
    windGusts?: number;
    enableDrag?: boolean;
    enableCooperative?: boolean;
    enableSwarm?: boolean;
    swarmFormation?: FormationType;
    swarmSpacing?: number;
    enableHmt?: boolean;
    hmtAuthorityLevel?: AuthorityLevel;
    enableDatalink?: boolean;
    enableTerrain?: boolean;
  }) => void;
  onStop: () => void;
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
  interceptGeometry: InterceptGeometry[] | null;
  threatAssessment: ThreatAssessment[] | null;
  onFetchInterceptGeometry: () => void;
  onFetchThreatAssessment: () => void;
  isRecording: boolean;
  recordings: RecordingMetadata[];
  onStartRecording: () => void;
  onStopRecording: () => void;
  onDeleteRecording: (id: string) => void;
  replayState: ReplayState | null;
  onStartReplay: (id: string) => void;
  onPauseReplay: () => void;
  onResumeReplay: () => void;
  onStopReplay: () => void;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
  wtaAlgorithms: WTAAlgorithm[];
  assignments: AssignmentResult | null;
  onFetchAssignments: (algorithm?: string) => void;
  environmentState: EnvironmentState | null;
  onFetchSensorTracks?: () => void;
  cooperativeState?: CooperativeState | null;
  onFetchCooperativeState?: () => void;
  onCreateEngagementZone?: (zone: EngagementZoneCreateRequest) => void;
  onDeleteEngagementZone?: (zoneId: string) => void;
  onAssignInterceptorToZone?: (interceptorId: string, zoneId: string) => void;
  onRequestHandoff?: (request: HandoffRequestCreate) => void;
  plannerMode?: string;
  onSetPlannerMode?: (mode: string) => void;
  plannedEntities?: { id: string; type: string; position: { x: number; y: number; z: number }; velocity: { x: number; y: number; z: number } }[];
  plannedZones?: { id: string; name: string; center: { x: number; y: number; z: number }; dimensions: { x: number; y: number; z: number }; color: string }[];
  onClearPlanner?: () => void;
  onRemovePlannedEntity?: (id: string) => void;
  mlStatus?: MLStatus | null;
  onFetchMLStatus?: () => void;
  swarmStatus?: SwarmStatus | null;
  formationTypes?: FormationInfo[];
  onFetchSwarmStatus?: () => void;
  onConfigureSwarm?: (config: Partial<SwarmConfig>) => void;
  onSetSwarmFormation?: (formation: FormationType) => void;
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

export function ControlPanel(props: ControlPanelProps) {
  const {
    connected, state, scenarios, guidanceLaws, evasionTypes,
    onStart, onStop, onRunMonteCarlo, onRunEnvelope,
    monteCarloLoading, envelopeLoading,
    interceptGeometry, threatAssessment,
    onFetchInterceptGeometry, onFetchThreatAssessment,
    isRecording, recordings, onStartRecording, onStopRecording, onDeleteRecording,
    replayState, onStartReplay, onPauseReplay, onResumeReplay, onStopReplay,
    showAdvanced, onToggleAdvanced,
    wtaAlgorithms, assignments, onFetchAssignments,
    environmentState, onFetchSensorTracks,
    cooperativeState, onFetchCooperativeState, onCreateEngagementZone,
    mlStatus, onFetchMLStatus,
    swarmStatus, formationTypes, onFetchSwarmStatus, onConfigureSwarm, onSetSwarmFormation,
    hmtStatus, authorityLevels, pendingActions,
    onFetchHMTStatus, onFetchPendingActions,
    onApproveAction, onRejectAction, onSetAuthorityLevel, onConfigureHMT,
  } = props;

  // ── Local form state ──────────────────────────────────────────
  const [selectedScenario, setSelectedScenario] = useState('head_on');
  const [selectedGuidance, setSelectedGuidance] = useState('proportional_nav');
  const [navConstant, setNavConstant] = useState(4.0);
  const [selectedEvasion, setSelectedEvasion] = useState('none');
  const [numInterceptors, setNumInterceptors] = useState(1);
  const [numTargets, setNumTargets] = useState(1);
  const [selectedWTA, setSelectedWTA] = useState('hungarian');
  const [windSpeed, setWindSpeed] = useState(0);
  const [windDirection, setWindDirection] = useState(0);
  const [windGusts, setWindGusts] = useState(0);
  const [enableDrag, setEnableDrag] = useState(false);
  const [enableCooperative, setEnableCooperative] = useState(false);
  const [enableSwarm, setEnableSwarm] = useState(false);
  const [swarmFormation, setSwarmFormation] = useState<FormationType>('line_abreast');
  const [swarmSpacing, setSwarmSpacing] = useState(100);
  const [enableHmt, setEnableHmt] = useState(false);
  const [hmtAuthorityLevel, setHmtAuthorityLevel] = useState<AuthorityLevel>('human_on_loop');
  const [enableDatalink] = useState(false);
  const [enableTerrain] = useState(false);

  const isRunning = state?.status === 'running';

  // ── Side-effects ──────────────────────────────────────────────
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      onFetchInterceptGeometry();
      onFetchThreatAssessment();
      if (numTargets > 1) onFetchAssignments(selectedWTA);
      if (onFetchSensorTracks) onFetchSensorTracks();
      if (onFetchCooperativeState && enableCooperative) onFetchCooperativeState();
      if (onFetchSwarmStatus && enableSwarm) onFetchSwarmStatus();
      if (enableHmt) {
        if (onFetchHMTStatus) onFetchHMTStatus();
        if (onFetchPendingActions) onFetchPendingActions();
      }
    }, 200);
    return () => clearInterval(interval);
  }, [isRunning, onFetchInterceptGeometry, onFetchThreatAssessment, onFetchAssignments, numTargets, selectedWTA, onFetchSensorTracks, onFetchCooperativeState, enableCooperative, onFetchSwarmStatus, enableSwarm, onFetchHMTStatus, onFetchPendingActions, enableHmt]);

  useEffect(() => {
    const scenario = scenarios[selectedScenario];
    if (scenario) {
      if (scenario.num_targets) setNumTargets(scenario.num_targets);
      if (scenario.evasion && scenario.evasion !== 'none') setSelectedEvasion(scenario.evasion);
    }
  }, [selectedScenario, scenarios]);

  // ── Handlers ──────────────────────────────────────────────────
  const handleLaunch = () => {
    onStart({
      scenario: selectedScenario, guidance: selectedGuidance, navConstant,
      evasion: selectedEvasion, numInterceptors, numTargets,
      wtaAlgorithm: selectedWTA,
      windSpeed, windDirection, windGusts, enableDrag, enableCooperative,
      enableSwarm, swarmFormation, swarmSpacing,
      enableHmt, hmtAuthorityLevel, enableDatalink, enableTerrain,
    });
  };

  // ── Render ────────────────────────────────────────────────────
  return (
    <>
      <MissionToolbar
        connected={connected} isRunning={isRunning}
        selectedScenario={selectedScenario} selectedGuidance={selectedGuidance}
        navConstant={navConstant} selectedEvasion={selectedEvasion}
        numInterceptors={numInterceptors} numTargets={numTargets} selectedWTA={selectedWTA}
        scenarios={scenarios} guidanceLaws={guidanceLaws}
        evasionTypes={evasionTypes} wtaAlgorithms={wtaAlgorithms}
        onScenario={setSelectedScenario} onGuidance={setSelectedGuidance}
        onNavConstant={setNavConstant} onEvasion={setSelectedEvasion}
        onNumInterceptors={setNumInterceptors} onNumTargets={setNumTargets} onWTA={setSelectedWTA}
        onLaunch={handleLaunch} onAbort={onStop}
        isRecording={isRecording}
        onToggleRecording={isRecording ? onStopRecording : onStartRecording}
        showAdvanced={showAdvanced} onToggleAdvanced={onToggleAdvanced}
      />

      <MissionStatusHUD
        state={state}
        interceptGeometry={interceptGeometry}
        threatAssessment={threatAssessment}
        assignments={assignments}
        replayState={replayState}
        onPauseReplay={onPauseReplay}
        onResumeReplay={onResumeReplay}
        onStopReplay={onStopReplay}
      />

      {showAdvanced && (
        <AdvancedPanel
          isRunning={isRunning}
          selectedScenario={selectedScenario} selectedGuidance={selectedGuidance}
          navConstant={navConstant} selectedEvasion={selectedEvasion}
          onRunMonteCarlo={onRunMonteCarlo} onRunEnvelope={onRunEnvelope}
          monteCarloLoading={monteCarloLoading} envelopeLoading={envelopeLoading}
          recordings={recordings} onStartReplay={onStartReplay} onDeleteRecording={onDeleteRecording}
          windSpeed={windSpeed} windDirection={windDirection} windGusts={windGusts}
          enableDrag={enableDrag}
          onWindSpeed={setWindSpeed} onWindDirection={setWindDirection}
          onWindGusts={setWindGusts} onEnableDrag={setEnableDrag}
          enableCooperative={enableCooperative} onEnableCooperative={setEnableCooperative}
          environmentState={environmentState}
          cooperativeState={cooperativeState} onCreateEngagementZone={onCreateEngagementZone}
          mlStatus={mlStatus} onFetchMLStatus={onFetchMLStatus}
          enableSwarm={enableSwarm} swarmFormation={swarmFormation} swarmSpacing={swarmSpacing}
          onEnableSwarm={setEnableSwarm} onSwarmFormation={setSwarmFormation} onSwarmSpacing={setSwarmSpacing}
          swarmStatus={swarmStatus} formationTypes={formationTypes}
          onFetchSwarmStatus={onFetchSwarmStatus} onConfigureSwarm={onConfigureSwarm}
          onSetSwarmFormation={onSetSwarmFormation}
          enableHmt={enableHmt} hmtAuthorityLevel={hmtAuthorityLevel}
          onEnableHmt={setEnableHmt}
          onHmtAuthorityLevel={(level) => {
            setHmtAuthorityLevel(level);
            if (isRunning && onSetAuthorityLevel) onSetAuthorityLevel(level);
          }}
          hmtStatus={hmtStatus} authorityLevels={authorityLevels} pendingActions={pendingActions}
          onFetchHMTStatus={onFetchHMTStatus} onFetchPendingActions={onFetchPendingActions}
          onApproveAction={onApproveAction} onRejectAction={onRejectAction}
          onSetAuthorityLevel={onSetAuthorityLevel} onConfigureHMT={onConfigureHMT}
        />
      )}
    </>
  );
}
