/**
 * Main App Component - Intercept Command & Control
 *
 * Layout:
 * - Top: Mission controls toolbar
 * - Status bar: System metrics + audio toggle
 * - Center: 3D visualization with telemetry overlay
 * - Bottom: Camera mode selector + replay theater
 * - Modals: Scenario briefing/debrief overlays
 */

import { useState, useEffect, useCallback } from 'react';
import { SimulationScene } from './components/Scene';
import { ControlPanel } from './components/ControlPanel';
import { HMTToast } from './components/HMTToast';
import { LaunchEventToast } from './components/LaunchEventToast';
import { MissionPlannerPanel } from './components/MissionPlannerPanel';
import { StatusBar } from './components/StatusBar';
import { TelemetryHUD } from './components/TelemetryHUD';
import { CameraModeSelector } from './components/CameraController';
import type { CameraMode } from './components/CameraController';
import { ReplayTheater } from './components/ReplayTheater';
import { ScenarioBriefing, ScenarioDebrief } from './components/ScenarioBriefing';
import { SplashScreen } from './components/SplashScreen';
import { NARRATIVE_SCENARIOS, calculateScore } from './data/scenarios';
import type { NarrativeScenario } from './data/scenarios';
import { useAudio } from './hooks/useAudio';
import { useSimulation } from './hooks/useSimulation';
import { useMissionPlanner } from './components/MissionPlanner';
import './App.css';

type ScenarioFlow = 'idle' | 'briefing' | 'running' | 'debrief';

function App() {
  const {
    connected,
    state,
    scenarios,
    guidanceLaws,
    evasionTypes,
    startRun,
    stopRun,
    runMonteCarlo,
    runEnvelopeAnalysis,
    monteCarloLoading,
    envelopeLoading,
    //
    interceptGeometry,
    fetchInterceptGeometry,
    threatAssessment,
    fetchThreatAssessment,
    isRecording,
    recordings,
    startRecording,
    stopRecording,
    deleteRecording,
    replayState,
    startReplay,
    pauseReplay,
    resumeReplay,
    stopReplay,
    seekReplay,
    //
    wtaAlgorithms,
    assignments,
    fetchAssignments,
    //
    environmentState,
    sensorTracks,
    fetchSensorTracks,
    // Cooperative Engagement
    cooperativeState,
    fetchCooperativeState,
    createEngagementZone,
    deleteEngagementZone,
    assignInterceptorToZone,
    requestHandoff,
    // ML
    mlStatus,
    fetchMLStatus,
    // Swarm
    swarmStatus,
    formationTypes,
    fetchSwarmStatus,
    configureSwarm,
    setSwarmFormation,
    // HMT
    hmtStatus,
    authorityLevels,
    pendingActions,
    fetchHMTStatus,
    fetchPendingActions,
    approveAction,
    rejectAction,
    setAuthorityLevel,
    configureHMT,
    // Launchers
    launchers,
  } = useSimulation();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('free');
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);

  // Audio system
  const audio = useAudio();

  // Scenario flow state machine
  const [scenarioFlow, setScenarioFlow] = useState<ScenarioFlow>('idle');
  const [activeScenario, setActiveScenario] = useState<NarrativeScenario | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ReturnType<typeof calculateScore> | null>(null);
  const [scenarioSimTime, setScenarioSimTime] = useState(0);
  const [scenarioMissDistance, setScenarioMissDistance] = useState(0);

  // Splash screen state
  const [showSplash, setShowSplash] = useState(true);

  // Replay theater state
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [autoCameraEnabled, setAutoCameraEnabled] = useState(false);

  // Mission Planner state
  const missionPlanner = useMissionPlanner();

  // Check if simulation is running
  const isRunning = state?.status === 'running';
  const isReplayActive = !!replayState;

  // Use planner's entity selection when not running, our own when running
  const activeSelectedId = isRunning ? selectedEntityId : missionPlanner.selectedEntityId;
  const activeSelectEntity = isRunning
    ? setSelectedEntityId
    : missionPlanner.selectEntity;

  // Get first geometry for telemetry HUD
  const firstGeometry = interceptGeometry && interceptGeometry.length > 0
    ? interceptGeometry[0]
    : null;

  // Audio triggers for simulation events
  useEffect(() => {
    if (isRunning) {
      audio.startAmbient();
    } else {
      audio.stopAmbient();
    }
  }, [isRunning, audio]);

  // Track intercepts for audio + scenario scoring
  const [prevInterceptCount, setPrevInterceptCount] = useState(0);
  useEffect(() => {
    const currentCount = state?.intercepted_pairs?.length || 0;
    if (currentCount > prevInterceptCount) {
      audio.playExplosion();
    }
    setPrevInterceptCount(currentCount);
  }, [state?.intercepted_pairs?.length, audio, prevInterceptCount]);

  // Scenario completion detection
  useEffect(() => {
    if (scenarioFlow !== 'running' || !activeScenario || !state) return;

    if (state.status === 'completed') {
      const simTime = state.sim_time || 0;
      const missDistance = firstGeometry?.predicted_miss_distance || 0;
      const interceptedCount = state.intercepted_pairs?.length || 0;
      const totalTargets = activeScenario.config.numTargets;

      const result = calculateScore(activeScenario, simTime, missDistance, interceptedCount, totalTargets);
      setScenarioResult(result);
      setScenarioSimTime(simTime);
      setScenarioMissDistance(missDistance);
      setScenarioFlow('debrief');
    }
  }, [state?.status, scenarioFlow, activeScenario, state, firstGeometry]);

  // Handle scenario briefing commence
  const handleScenarioCommence = useCallback(async () => {
    if (!activeScenario) return;
    audio.playUIClick();

    const config = activeScenario.config;
    await startRun({
      scenario: config.scenario,
      guidance: config.guidance,
      evasion: config.evasion,
      navConstant: config.navConstant,
      numInterceptors: config.numInterceptors,
      numTargets: config.numTargets,
      enableCooperative: config.enableCooperative,
      enableSwarm: config.enableSwarm,
      enableHmt: config.enableHMT,
      windSpeed: config.windSpeed,
      windDirection: config.windDirection,
      enableDrag: config.enableDrag,
    });

    setScenarioFlow('running');
  }, [activeScenario, startRun, audio]);

  // Select and open a scenario
  const handleSelectScenario = useCallback((scenario: NarrativeScenario) => {
    audio.playUIClick();
    setActiveScenario(scenario);
    setScenarioResult(null);
    setScenarioFlow('briefing');
  }, [audio]);

  // Handle launch with custom entities
  const handleLaunch = async (options: Parameters<typeof startRun>[0]) => {
    audio.playLaunch();

    if (missionPlanner.plannedEntities.length > 0) {
      const regularEntities = missionPlanner.plannedEntities.filter(e => e.type !== 'launcher');
      const launcherEntities = missionPlanner.plannedEntities.filter(e => e.type === 'launcher');

      const customEntities = regularEntities.map(e => ({
        id: e.id,
        type: e.type,
        position: e.position,
        velocity: e.velocity,
      }));

      const customLaunchers = launcherEntities.map(e => ({
        id: e.id,
        position: e.position,
        detection_range: e.launcherConfig?.detectionRange || 5000,
        num_missiles: e.launcherConfig?.numMissiles || 4,
        launch_mode: e.launcherConfig?.launchMode || 'auto',
      }));

      const customZones = missionPlanner.plannedZones.map(z => ({
        id: z.id,
        name: z.name,
        center: z.center,
        dimensions: z.dimensions,
        color: z.color,
      }));

      await startRun({
        ...options,
        customEntities,
        customLaunchers: customLaunchers.length > 0 ? customLaunchers : undefined,
        customZones: options.enableCooperative ? customZones : undefined,
      });

      missionPlanner.setMode('view');
    } else {
      await startRun(options);
    }
  };

  // Replay theater handlers
  const handleReplaySeek = useCallback((tick: number) => {
    seekReplay?.(tick);
  }, [seekReplay]);

  const handleReplaySetSpeed = useCallback((speed: number) => {
    setReplaySpeed(speed);
  }, []);

  const handleToggleAutoCamera = useCallback(() => {
    setAutoCameraEnabled(prev => {
      const next = !prev;
      if (next) {
        setCameraMode('cinematic');
      } else {
        setCameraMode('free');
      }
      return next;
    });
  }, []);

  // Replay progress for cinematic camera
  const replayProgress = replayState
    ? (replayState.current_tick || 0) / Math.max(replayState.total_ticks || 1, 1)
    : 0;

  return (
    <div className="app">
      {/* Scenario Briefing Modal */}
      {scenarioFlow === 'briefing' && activeScenario && (
        <ScenarioBriefing
          scenario={activeScenario}
          onCommence={handleScenarioCommence}
          onBack={() => { setScenarioFlow('idle'); setActiveScenario(null); }}
        />
      )}

      {/* Scenario Debrief Modal */}
      {scenarioFlow === 'debrief' && activeScenario && scenarioResult && (
        <ScenarioDebrief
          scenario={activeScenario}
          result={scenarioResult}
          simTime={scenarioSimTime}
          missDistance={scenarioMissDistance}
          onReplay={() => {
            // Start replay of the last recording
            if (recordings && recordings.length > 0) {
              startReplay(recordings[recordings.length - 1].recording_id, { speed_multiplier: replaySpeed, start_tick: 0 });
            }
            setScenarioFlow('idle');
          }}
          onNext={() => {
            // Find next scenario
            const idx = NARRATIVE_SCENARIOS.findIndex(s => s.id === activeScenario.id);
            const next = NARRATIVE_SCENARIOS[(idx + 1) % NARRATIVE_SCENARIOS.length];
            handleSelectScenario(next);
          }}
          onBack={() => { setScenarioFlow('idle'); setActiveScenario(null); }}
        />
      )}

      {showSplash && (
        <SplashScreen
          connected={connected}
          onComplete={() => setShowSplash(false)}
        />
      )}

      <header className="app-header">
        <div className="header-left">
          <div className="header-brand">
            <h1>Intercept</h1>
            <span className="header-subtitle">AIR DEFENSE COMMAND</span>
          </div>
          <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {connected ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
        <div className="header-center">
          {/* Scenario quick-select buttons */}
          <div className="scenario-buttons">
            {NARRATIVE_SCENARIOS.map(s => (
              <button
                key={s.id}
                className={`scenario-btn scenario-${s.difficulty.toLowerCase()} ${activeScenario?.id === s.id ? 'active' : ''}`}
                onClick={() => handleSelectScenario(s)}
                title={`${s.name} - ${s.difficulty}`}
                disabled={isRunning}
              >
                {s.codename.split(' ')[0]}
              </button>
            ))}
          </div>
          <ControlPanel
            connected={connected}
            state={state}
            scenarios={scenarios}
            guidanceLaws={guidanceLaws}
            evasionTypes={evasionTypes}
            onStart={handleLaunch}
            onStop={stopRun}
            onRunMonteCarlo={runMonteCarlo}
            onRunEnvelope={runEnvelopeAnalysis}
            monteCarloLoading={monteCarloLoading}
            envelopeLoading={envelopeLoading}
            interceptGeometry={interceptGeometry}
            threatAssessment={threatAssessment}
            onFetchInterceptGeometry={fetchInterceptGeometry}
            onFetchThreatAssessment={fetchThreatAssessment}
            isRecording={isRecording}
            recordings={recordings}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onDeleteRecording={deleteRecording}
            replayState={replayState}
            onStartReplay={startReplay}
            onPauseReplay={pauseReplay}
            onResumeReplay={resumeReplay}
            onStopReplay={stopReplay}
            showAdvanced={showAdvanced}
            onToggleAdvanced={() => setShowAdvanced(!showAdvanced)}
            //
            wtaAlgorithms={wtaAlgorithms}
            assignments={assignments}
            onFetchAssignments={fetchAssignments}
            //
            environmentState={environmentState}
            onFetchSensorTracks={fetchSensorTracks}
            // Cooperative Engagement
            cooperativeState={cooperativeState}
            onFetchCooperativeState={fetchCooperativeState}
            onCreateEngagementZone={createEngagementZone}
            onDeleteEngagementZone={deleteEngagementZone}
            onAssignInterceptorToZone={assignInterceptorToZone}
            onRequestHandoff={requestHandoff}
            // Mission Planner
            plannerMode={missionPlanner.mode}
            onSetPlannerMode={(mode: string) => missionPlanner.setMode(mode as 'view' | 'interceptor' | 'target' | 'launcher' | 'zone')}
            plannedEntities={missionPlanner.plannedEntities}
            plannedZones={missionPlanner.plannedZones}
            onClearPlanner={missionPlanner.clearAll}
            onRemovePlannedEntity={missionPlanner.removeEntity}
            // ML
            mlStatus={mlStatus}
            onFetchMLStatus={fetchMLStatus}
            // Swarm
            swarmStatus={swarmStatus}
            formationTypes={formationTypes}
            onFetchSwarmStatus={fetchSwarmStatus}
            onConfigureSwarm={configureSwarm}
            onSetSwarmFormation={setSwarmFormation}
            // HMT
            hmtStatus={hmtStatus}
            authorityLevels={authorityLevels}
            pendingActions={pendingActions}
            onFetchHMTStatus={fetchHMTStatus}
            onFetchPendingActions={fetchPendingActions}
            onApproveAction={approveAction}
            onRejectAction={rejectAction}
            onSetAuthorityLevel={setAuthorityLevel}
            onConfigureHMT={configureHMT}
          />
        </div>
      </header>

      <StatusBar
        state={state}
        connected={connected}
        isRecording={isRecording}
        muted={audio.muted}
        onToggleMute={audio.toggleMute}
      />

      {/* HMT Toast - Floating notification for pending actions */}
      <HMTToast
        pendingActions={pendingActions}
        onApprove={approveAction}
        onReject={rejectAction}
        enabled={isRunning}
      />

      {/* Launch Event Toast - Command Center alerts */}
      <LaunchEventToast
        launchers={launchers}
        enabled={isRunning}
      />

      <main className="app-main">
        <div className={`scene-container ${!isRunning && !isReplayActive ? 'with-planner' : ''}`}>
          <SimulationScene
            state={state}
            interceptGeometry={interceptGeometry}
            assignments={assignments}
            sensorTracks={sensorTracks}
            cooperativeState={cooperativeState}
            launchers={launchers}
            // Camera & selection
            cameraMode={cameraMode}
            selectedEntityId={activeSelectedId}
            onSelectEntity={activeSelectEntity}
            replayProgress={autoCameraEnabled ? replayProgress : undefined}
            // Mission Planner
            plannerMode={missionPlanner.mode}
            plannedEntities={missionPlanner.plannedEntities}
            plannedZones={missionPlanner.plannedZones}
            onAddEntity={missionPlanner.addEntity}
            onUpdateEntity={missionPlanner.updateEntity}
            onRemoveEntity={missionPlanner.removeEntity}
            onAddZone={missionPlanner.addZone}
            onUpdateZone={missionPlanner.updateZone}
            onRemoveZone={missionPlanner.removeZone}
            showGrid={missionPlanner.showGrid}
            snapToGrid={missionPlanner.snapToGrid}
          />

          {/* Telemetry HUD overlay on scene */}
          <TelemetryHUD
            state={state}
            selectedEntityId={activeSelectedId}
            interceptGeometry={firstGeometry}
          />

          {/* Camera mode selector */}
          {!isReplayActive && (
            <CameraModeSelector
              mode={cameraMode}
              onSetMode={setCameraMode}
              hasSelection={!!activeSelectedId}
            />
          )}
        </div>

        {/* Replay Theater - shown during replay */}
        {isReplayActive && replayState && (
          <ReplayTheater
            currentTick={replayState.current_tick || 0}
            totalTicks={replayState.total_ticks || 0}
            isPlaying={replayState.is_playing && !replayState.is_paused}
            speed={replaySpeed}
            onSeek={handleReplaySeek}
            onPause={pauseReplay}
            onResume={resumeReplay}
            onStop={stopReplay}
            onSetSpeed={handleReplaySetSpeed}
            autoCameraEnabled={autoCameraEnabled}
            onToggleAutoCamera={handleToggleAutoCamera}
          />
        )}
      </main>

      {/* Mission Planner Sidebar - shown when not running and not in replay */}
      {!isRunning && !isReplayActive && (
        <MissionPlannerPanel
          mode={missionPlanner.mode}
          onSetMode={missionPlanner.setMode}
          plannedEntities={missionPlanner.plannedEntities}
          plannedZones={missionPlanner.plannedZones}
          selectedEntityId={missionPlanner.selectedEntityId}
          onSelectEntity={missionPlanner.selectEntity}
          onUpdateEntity={missionPlanner.updateEntity}
          onRemoveEntity={missionPlanner.removeEntity}
          onUpdateZone={missionPlanner.updateZone}
          onRemoveZone={missionPlanner.removeZone}
          onClearAll={missionPlanner.clearAll}
          showGrid={missionPlanner.showGrid}
          onToggleGrid={() => missionPlanner.setShowGrid(!missionPlanner.showGrid)}
          snapToGrid={missionPlanner.snapToGrid}
          onToggleSnap={() => missionPlanner.setSnapToGrid(!missionPlanner.snapToGrid)}
          isSimRunning={isRunning}
        />
      )}
    </div>
  );
}

export default App;
