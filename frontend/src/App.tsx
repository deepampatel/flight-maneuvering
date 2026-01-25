/**
 * Main App Component - Mission Control UI
 *
 * Layout:
 * - Top: Mission controls toolbar
 * - Center: 3D visualization (full width)
 * - Bottom: Compact telemetry bar + expandable panels
 */

import { useState } from 'react';
import { SimulationScene } from './components/Scene';
import { ControlPanel } from './components/ControlPanel';
import { HMTToast } from './components/HMTToast';
import { LaunchEventToast } from './components/LaunchEventToast';
import { MissionPlannerPanel } from './components/MissionPlannerPanel';
import { useSimulation } from './hooks/useSimulation';
import { useMissionPlanner } from './components/MissionPlanner';
import './App.css';

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

  // Mission Planner state
  const missionPlanner = useMissionPlanner();

  // Check if simulation is running
  const isRunning = state?.status === 'running';

  // Handle launch with custom entities
  const handleLaunch = async (options: Parameters<typeof startRun>[0]) => {
    // If we have planned entities, use them
    if (missionPlanner.plannedEntities.length > 0) {
      // Separate launchers from regular entities
      const regularEntities = missionPlanner.plannedEntities.filter(e => e.type !== 'launcher');
      const launcherEntities = missionPlanner.plannedEntities.filter(e => e.type === 'launcher');

      const customEntities = regularEntities.map(e => ({
        id: e.id,
        type: e.type,
        position: e.position,
        velocity: e.velocity,
      }));

      // Format launchers for the backend
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

      // Switch back to view mode after launch
      missionPlanner.setMode('view');
    } else {
      await startRun(options);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>Intercept</h1>
          <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {connected ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
        <div className="header-center">
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

      {/* HMT Toast - Floating notification for pending actions */}
      <HMTToast
        pendingActions={pendingActions}
        onApprove={approveAction}
        onReject={rejectAction}
        enabled={isRunning}
      />

      {/* Launch Event Toast - Command Center alerts for launcher events */}
      <LaunchEventToast
        launchers={launchers}
        enabled={isRunning}
      />

      <main className="app-main">
        <div className={`scene-container ${!isRunning ? 'with-planner' : ''}`}>
          <SimulationScene
            state={state}
            interceptGeometry={interceptGeometry}
            assignments={assignments}
            sensorTracks={sensorTracks}
            cooperativeState={cooperativeState}
            launchers={launchers}
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
            selectedEntityId={missionPlanner.selectedEntityId}
            onSelectEntity={missionPlanner.selectEntity}
            showGrid={missionPlanner.showGrid}
            snapToGrid={missionPlanner.snapToGrid}
          />
        </div>
      </main>


      {/* Mission Planner Sidebar - shown when not running */}
      {!isRunning && (
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
