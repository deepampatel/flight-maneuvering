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
import { TelemetryHUD } from './components/TelemetryHUD';
import { CameraModeSelector } from './components/CameraController';
import type { CameraMode } from './components/CameraController';
import { ReplayTheater } from './components/ReplayTheater';
import { ScenarioBriefing, ScenarioDebrief } from './components/ScenarioBriefing';
import { SplashScreen } from './components/SplashScreen';
import { WelcomeModal } from './components/WelcomeModal';
import { KeyboardShortcutsPanel } from './components/KeyboardShortcutsPanel';
import { RadarSweep } from './components/RadarSweep';
import { EngagementTimeline } from './components/EngagementTimeline';
import { LayerDiagram } from './components/LayerDiagram';
import { ScenarioBuilder } from './components/ScenarioBuilder';
import { NARRATIVE_SCENARIOS, calculateScore } from './data/scenarios';
import type { NarrativeScenario } from './data/scenarios';
import type { ScenarioBuilderState } from './types';
import { useAudio } from './hooks/useAudio';
import { useSimulation } from './hooks/useSimulation';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
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
  const [builderOpen, setBuilderOpen] = useState(true);
  const [plannerOpen, setPlannerOpen] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('free');
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [showRadar, setShowRadar] = useState(false);
  const [showLayers, setShowLayers] = useState(false);

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

  // Scene placement bridge (ScenarioBuilder → 3D scene → ScenarioBuilder)
  const [scenePlacement, setScenePlacement] = useState<{
    type: 'battery' | 'protected_area';
    position: { x: number; y: number; z: number };
  } | null>(null);

  const handleRequestPlacement = useCallback((type: 'battery' | 'protected_area') => {
    // Set mission planner to the placement mode so the 3D scene accepts clicks
    missionPlanner.setMode(type);
    // Auto-open builder so user can see the result
    setBuilderOpen(true);
  }, [missionPlanner]);

  const handleScenePlacement = useCallback((type: 'battery' | 'protected_area', position: { x: number; y: number; z: number }) => {
    setScenePlacement({ type, position });
    // Reset mode back to view after placement
    missionPlanner.setMode('view');
  }, [missionPlanner]);

  const handlePlacementConsumed = useCallback(() => {
    setScenePlacement(null);
  }, []);

  // Builder state for 3D preview (batteries/areas shown before launch)
  const [builderPreview, setBuilderPreview] = useState<ScenarioBuilderState | null>(null);

  // Keyboard shortcuts
  const isRunning = state?.status === 'running';
  const { showHelp, setShowHelp, shortcuts } = useKeyboardShortcuts({
    isRunning,
    onLaunch: () => {}, // Launch is handled via ControlPanel
    onStop: stopRun,
    onToggleRecording: () => isRecording ? stopRecording() : startRecording(),
    onCameraMode: (mode) => setCameraMode(mode as CameraMode),
    onToggleAdvanced: () => setShowAdvanced(prev => !prev),
  });

  // Check if simulation is running
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

  // Scenario Builder launch handler
  const handleBuilderLaunch = useCallback(async (builderState: ScenarioBuilderState, presetScenario: string | null) => {
    audio.playLaunch();

    // Always send explicit arrays — "what you see in the builder = what runs"
    const customBatteries = builderState.batteries.map(b => ({
      id: b.id,
      name: b.name,
      tier: b.tier,
      position: b.position,
      radar_range: b.radar_range,
      radar_sector: b.radar_sector,
      num_launchers: b.num_launchers,
      missiles_per_launcher: b.missiles_per_launcher,
      max_simultaneous: b.max_simultaneous,
      min_range: b.min_range,
      max_range: b.max_range,
      launch_speed: b.launch_speed,
      launch_elevation: b.launch_elevation,
      min_altitude: b.min_altitude,
      protected_area_ids: b.protected_area_ids,
    }));

    const customWaves = builderState.waves.map(w => ({
      delay: w.delay,
      threat_type: w.threat_type,
      count: w.count,
      spawn_bearing: w.spawn_bearing,
      spawn_range: w.spawn_range,
      spawn_altitude: w.spawn_altitude,
      spacing: w.spacing,
      salvo_interval: w.salvo_interval,
      decoy_fraction: w.decoy_fraction,
    }));

    const customProtectedAreas = builderState.protectedAreas.map(a => ({
      id: a.id,
      name: a.name,
      center: a.center,
      radius: a.radius,
      priority: a.priority,
    }));

    await startRun({
      scenario: presetScenario || 'head_on',
      guidance: builderState.guidance,
      navConstant: builderState.navConstant,
      evasion: builderState.evasion,
      numInterceptors: builderState.numInterceptors,
      numTargets: builderState.numTargets,
      wtaAlgorithm: builderState.wtaAlgorithm,
      killRadius: builderState.killRadius,
      windSpeed: builderState.windSpeed,
      windDirection: builderState.windDirection,
      windGusts: builderState.windGusts,
      enableDrag: builderState.enableDrag,
      enableTerrain: builderState.enableTerrain,
      enableDatalink: builderState.enableDatalink,
      enableCooperative: builderState.enableCooperative,
      enableSwarm: builderState.enableSwarm,
      enableHmt: builderState.enableHmt,
      hmtAuthorityLevel: builderState.hmtAuthorityLevel as 'full_auto' | 'human_on_loop' | 'human_in_loop' | 'manual',
      maxTime: builderState.maxTime,
      customBatteries,
      customWaves,
      customProtectedAreas,
    });
  }, [startRun, audio]);

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

      {/* Welcome onboarding (first visit) */}
      {!showSplash && <WelcomeModal />}

      {/* Keyboard shortcuts overlay */}
      {showHelp && (
        <KeyboardShortcutsPanel shortcuts={shortcuts} onClose={() => setShowHelp(false)} />
      )}

      <header className="app-header">
        <div className="header-left">
          <h1 className="header-brand-text">Intercept</h1>
        </div>
        <div className="header-toggles">
          <button
            className={`header-toggle-btn ${builderOpen ? 'active' : ''}`}
            onClick={() => setBuilderOpen(p => !p)}
            title="Scenario Builder (B)"
          >
            <svg className="toggle-icon" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
              <path d="M1 3h14v1.5H1zm0 4h14v1.5H1zm0 4h10v1.5H1z"/>
            </svg>
            <span className="toggle-label">Builder</span>
          </button>
          <button
            className={`header-toggle-btn ${plannerOpen ? 'active' : ''}`}
            onClick={() => setPlannerOpen(p => !p)}
            title="Mission Planner (P)"
          >
            <svg className="toggle-icon" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
              <path d="M8 1l2 5h5l-4 3.5 1.5 5L8 11.5 3.5 14.5 5 9.5 1 6h5z"/>
            </svg>
            <span className="toggle-label">Planner</span>
          </button>
          <button
            className={`header-toggle-btn ${showAdvanced ? 'active' : ''}`}
            onClick={() => setShowAdvanced(p => !p)}
            title="Analysis Panel (A)"
          >
            <svg className="toggle-icon" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
              <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 1.5a5.5 5.5 0 110 11 5.5 5.5 0 010-11zM7.25 4v5h1.5V4h-1.5zM7.25 10.5v1.5h1.5v-1.5h-1.5z"/>
            </svg>
            <span className="toggle-label">Analysis</span>
          </button>
        </div>
        <div className="header-sim-status">
          {isRunning && (
            <>
              <span className="sim-status-chip">T+{(state?.sim_time || 0).toFixed(1)}s</span>
              <span className="sim-status-chip">{state?.entities?.length || 0} entities</span>
              <span className={`sim-status-chip status-${(state?.status || 'idle').toLowerCase()}`}>
                {(state?.status || 'IDLE').toUpperCase()}
              </span>
            </>
          )}
        </div>
        <div className="header-right">
          {isRecording && <span className="header-rec-badge">REC</span>}
          <button className="header-icon-btn" onClick={audio.toggleMute}
            title={audio.muted ? 'Unmute audio' : 'Mute audio'}>
            {audio.muted ? '\u{1F507}' : '\u{1F50A}'}
          </button>
          <button className="header-icon-btn" onClick={() => setShowHelp(true)} title="Keyboard shortcuts (?)">
            ?
          </button>
          <div className={`connection-dot ${connected ? 'online' : 'offline'}`}
            title={connected ? 'Connected' : 'Disconnected'} />
        </div>
      </header>

      {/* ControlPanel — hidden toolbar, keeps polling + MissionStatusHUD + AdvancedPanel */}
      <ControlPanel
        hideToolbar
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
        wtaAlgorithms={wtaAlgorithms}
        assignments={assignments}
        onFetchAssignments={fetchAssignments}
        environmentState={environmentState}
        onFetchSensorTracks={fetchSensorTracks}
        cooperativeState={cooperativeState}
        onFetchCooperativeState={fetchCooperativeState}
        onCreateEngagementZone={createEngagementZone}
        onDeleteEngagementZone={deleteEngagementZone}
        onAssignInterceptorToZone={assignInterceptorToZone}
        onRequestHandoff={requestHandoff}
        plannerMode={missionPlanner.mode}
        onSetPlannerMode={(mode: string) => missionPlanner.setMode(mode as 'view' | 'interceptor' | 'target' | 'launcher' | 'zone')}
        plannedEntities={missionPlanner.plannedEntities}
        plannedZones={missionPlanner.plannedZones}
        onClearPlanner={missionPlanner.clearAll}
        onRemovePlannedEntity={missionPlanner.removeEntity}
        mlStatus={mlStatus}
        onFetchMLStatus={fetchMLStatus}
        swarmStatus={swarmStatus}
        formationTypes={formationTypes}
        onFetchSwarmStatus={fetchSwarmStatus}
        onConfigureSwarm={configureSwarm}
        onSetSwarmFormation={setSwarmFormation}
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

      {/* Scenario Builder Drawer */}
      <ScenarioBuilder
        open={builderOpen}
        onToggle={() => setBuilderOpen(prev => !prev)}
        isRunning={isRunning}
        connected={connected}
        isRecording={isRecording}
        onLaunch={handleBuilderLaunch}
        onStop={stopRun}
        onToggleRecording={() => isRecording ? stopRecording() : startRecording()}
        onRequestPlacement={handleRequestPlacement}
        placementResult={scenePlacement}
        onPlacementConsumed={handlePlacementConsumed}
        onStateChange={setBuilderPreview}
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
        <div className={`scene-container ${!isRunning && !isReplayActive && plannerOpen ? 'with-planner' : ''}`}>
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
            onScenePlacement={handleScenePlacement}
            previewBatteries={!isRunning ? builderPreview?.batteries : undefined}
            previewProtectedAreas={!isRunning ? builderPreview?.protectedAreas : undefined}
          />

          {/* Telemetry HUD overlay on scene */}
          <TelemetryHUD
            state={state}
            selectedEntityId={activeSelectedId}
            interceptGeometry={firstGeometry}
          />

          {/* Radar PPI overlay — togglable */}
          {isRunning && showRadar && <RadarSweep state={state} />}

          {/* Layer Diagram — togglable */}
          {isRunning && showLayers && <LayerDiagram state={state} />}

          {/* Overlay toggles — bottom-right of scene */}
          {isRunning && (
            <div className="scene-overlay-toggles">
              <button
                className={`overlay-toggle-btn ${showRadar ? 'active' : ''}`}
                onClick={() => setShowRadar(r => !r)}
                title="Toggle Radar PPI"
              >
                Radar
              </button>
              <button
                className={`overlay-toggle-btn ${showLayers ? 'active' : ''}`}
                onClick={() => setShowLayers(l => !l)}
                title="Toggle Layer Diagram"
              >
                Layers
              </button>
            </div>
          )}

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

      {/* Engagement Timeline — bottom strip */}
      {isRunning && <EngagementTimeline state={state} />}

      {/* Mission Planner Sidebar - toggleable via header */}
      {!isRunning && !isReplayActive && plannerOpen && (
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
