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
    // Phase 4
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
    // Phase 5
    wtaAlgorithms,
    assignments,
    fetchAssignments,
    // Phase 6
    environmentState,
    sensorTracks,
    fetchSensorTracks,
    // Phase 6: Cooperative Engagement
    cooperativeState,
    fetchCooperativeState,
    createEngagementZone,
    deleteEngagementZone,
    assignInterceptorToZone,
    requestHandoff,
    // Phase 6.4: ML
    mlStatus,
    fetchMLStatus,
    // Phase 7: Swarm
    swarmStatus,
    formationTypes,
    fetchSwarmStatus,
    configureSwarm,
    setSwarmFormation,
    // Phase 7: HMT
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
          <h1>AIR DOMINANCE</h1>
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
            // Phase 5
            wtaAlgorithms={wtaAlgorithms}
            assignments={assignments}
            onFetchAssignments={fetchAssignments}
            // Phase 6
            environmentState={environmentState}
            onFetchSensorTracks={fetchSensorTracks}
            // Phase 6: Cooperative Engagement
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
            // Phase 6.4: ML
            mlStatus={mlStatus}
            onFetchMLStatus={fetchMLStatus}
            // Phase 7: Swarm
            swarmStatus={swarmStatus}
            formationTypes={formationTypes}
            onFetchSwarmStatus={fetchSwarmStatus}
            onConfigureSwarm={configureSwarm}
            onSetSwarmFormation={setSwarmFormation}
            // Phase 7: HMT
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
        <div className="scene-container">
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

      {/* Entity Property Panel - shown when entity is selected */}
      {!isRunning && missionPlanner.selectedEntityId && (() => {
        const selectedEntity = missionPlanner.plannedEntities.find(e => e.id === missionPlanner.selectedEntityId);
        const selectedZone = missionPlanner.plannedZones.find(z => z.id === missionPlanner.selectedEntityId);

        if (selectedEntity) {
          // Check if it's a launcher
          if (selectedEntity.type === 'launcher') {
            return (
              <div className="property-panel launcher">
                <div className="property-panel-header">
                  <span className="entity-type-badge launcher">&#9651;</span>
                  <span className="entity-id">{selectedEntity.id}</span>
                  <button
                    className="panel-close-btn"
                    onClick={() => missionPlanner.selectEntity(null)}
                    title="Close (Esc)"
                  >
                    &#10005;
                  </button>
                </div>

                <div className="property-section">
                  <div className="property-section-title">POSITION (m)</div>
                  <div className="property-grid">
                    <label>X</label>
                    <input
                      type="number"
                      value={Math.round(selectedEntity.position.x)}
                      onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                        position: { ...selectedEntity.position, x: parseFloat(e.target.value) || 0 }
                      })}
                    />
                    <label>Y</label>
                    <input
                      type="number"
                      value={Math.round(selectedEntity.position.y)}
                      onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                        position: { ...selectedEntity.position, y: parseFloat(e.target.value) || 0 }
                      })}
                    />
                  </div>
                </div>

                <div className="property-section">
                  <div className="property-section-title">LAUNCHER CONFIG</div>
                  <div className="property-grid">
                    <label>Range (m)</label>
                    <input
                      type="number"
                      value={selectedEntity.launcherConfig?.detectionRange || 5000}
                      onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                        launcherConfig: {
                          ...selectedEntity.launcherConfig!,
                          detectionRange: Math.max(1000, parseFloat(e.target.value) || 5000)
                        }
                      })}
                    />
                    <label>Missiles</label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={selectedEntity.launcherConfig?.numMissiles || 4}
                      onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                        launcherConfig: {
                          ...selectedEntity.launcherConfig!,
                          numMissiles: Math.max(1, Math.min(20, parseInt(e.target.value) || 4))
                        }
                      })}
                    />
                  </div>
                  <div className="property-summary">
                    <span>Range: {((selectedEntity.launcherConfig?.detectionRange || 5000) / 1000).toFixed(1)} km</span>
                  </div>
                </div>

                <div className="property-actions">
                  <button
                    className="property-action-btn delete"
                    onClick={() => missionPlanner.removeEntity(selectedEntity.id)}
                  >
                    Delete Launcher
                  </button>
                </div>
              </div>
            );
          }

          const speed = Math.sqrt(
            selectedEntity.velocity.x ** 2 +
            selectedEntity.velocity.y ** 2 +
            selectedEntity.velocity.z ** 2
          );
          const heading = Math.atan2(selectedEntity.velocity.y, selectedEntity.velocity.x) * (180 / Math.PI);

          return (
            <div className="property-panel">
              <div className="property-panel-header">
                <span className={`entity-type-badge ${selectedEntity.type}`}>
                  {selectedEntity.type === 'interceptor' ? '&#9650;' : '&#9679;'}
                </span>
                <span className="entity-id">{selectedEntity.id}</span>
                <button
                  className="panel-close-btn"
                  onClick={() => missionPlanner.selectEntity(null)}
                  title="Close (Esc)"
                >
                  &#10005;
                </button>
              </div>

              <div className="property-section">
                <div className="property-section-title">POSITION (m)</div>
                <div className="property-grid">
                  <label>X</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.position.x)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      position: { ...selectedEntity.position, x: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Y</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.position.y)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      position: { ...selectedEntity.position, y: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Alt</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.position.z)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      position: { ...selectedEntity.position, z: parseFloat(e.target.value) || 0 }
                    })}
                  />
                </div>
              </div>

              <div className="property-section">
                <div className="property-section-title">VELOCITY (m/s)</div>
                <div className="property-grid">
                  <label>Vx</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.velocity.x)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      velocity: { ...selectedEntity.velocity, x: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Vy</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.velocity.y)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      velocity: { ...selectedEntity.velocity, y: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Vz</label>
                  <input
                    type="number"
                    value={Math.round(selectedEntity.velocity.z)}
                    onChange={(e) => missionPlanner.updateEntity(selectedEntity.id, {
                      velocity: { ...selectedEntity.velocity, z: parseFloat(e.target.value) || 0 }
                    })}
                  />
                </div>
                <div className="property-summary">
                  <span>Speed: {speed.toFixed(0)} m/s</span>
                  <span>Heading: {heading.toFixed(0)}°</span>
                </div>
              </div>

              <div className="property-actions">
                <button
                  className="property-action-btn delete"
                  onClick={() => missionPlanner.removeEntity(selectedEntity.id)}
                >
                  Delete Entity
                </button>
              </div>
            </div>
          );
        }

        if (selectedZone) {
          return (
            <div className="property-panel zone">
              <div className="property-panel-header">
                <span className="entity-type-badge zone">&#9634;</span>
                <span className="entity-id">{selectedZone.name}</span>
                <button
                  className="panel-close-btn"
                  onClick={() => missionPlanner.selectEntity(null)}
                  title="Close (Esc)"
                >
                  &#10005;
                </button>
              </div>

              <div className="property-section">
                <div className="property-section-title">CENTER (m)</div>
                <div className="property-grid">
                  <label>X</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.center.x)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      center: { ...selectedZone.center, x: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Y</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.center.y)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      center: { ...selectedZone.center, y: parseFloat(e.target.value) || 0 }
                    })}
                  />
                  <label>Alt</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.center.z)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      center: { ...selectedZone.center, z: parseFloat(e.target.value) || 0 }
                    })}
                  />
                </div>
              </div>

              <div className="property-section">
                <div className="property-section-title">DIMENSIONS (m)</div>
                <div className="property-grid">
                  <label>Width</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.dimensions.x)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      dimensions: { ...selectedZone.dimensions, x: Math.max(100, parseFloat(e.target.value) || 100) }
                    })}
                  />
                  <label>Depth</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.dimensions.y)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      dimensions: { ...selectedZone.dimensions, y: Math.max(100, parseFloat(e.target.value) || 100) }
                    })}
                  />
                  <label>Height</label>
                  <input
                    type="number"
                    value={Math.round(selectedZone.dimensions.z)}
                    onChange={(e) => missionPlanner.updateZone(selectedZone.id, {
                      dimensions: { ...selectedZone.dimensions, z: Math.max(100, parseFloat(e.target.value) || 100) }
                    })}
                  />
                </div>
                <div className="property-summary">
                  <span>Area: {((selectedZone.dimensions.x * selectedZone.dimensions.y) / 1e6).toFixed(2)} km²</span>
                </div>
              </div>

              <div className="property-actions">
                <button
                  className="property-action-btn delete"
                  onClick={() => missionPlanner.removeZone(selectedZone.id)}
                >
                  Delete Zone
                </button>
              </div>
            </div>
          );
        }

        return null;
      })()}

      {/* Mission Planner Toolbar - shown when not running */}
      {!isRunning && (
        <div className="planner-toolbar">
          {/* Mode Selection */}
          <div className="planner-section">
            <div className="planner-section-label">MODE</div>
            <div className="planner-modes">
              <button
                className={`planner-btn ${missionPlanner.mode === 'view' ? 'active' : ''}`}
                onClick={() => missionPlanner.setMode('view')}
                title="View Mode (V) - Select and edit entities"
              >
                <span className="btn-icon">&#9670;</span>
                <span className="btn-text">Select</span>
              </button>
              <button
                className={`planner-btn interceptor ${missionPlanner.mode === 'interceptor' ? 'active' : ''}`}
                onClick={() => missionPlanner.setMode('interceptor')}
                title="Place Interceptor (I) - Click & drag to place"
              >
                <span className="btn-icon">&#9650;</span>
                <span className="btn-text">Interceptor</span>
              </button>
              <button
                className={`planner-btn target ${missionPlanner.mode === 'target' ? 'active' : ''}`}
                onClick={() => missionPlanner.setMode('target')}
                title="Place Target (T) - Click & drag to place"
              >
                <span className="btn-icon">&#9679;</span>
                <span className="btn-text">Target</span>
              </button>
              <button
                className={`planner-btn launcher ${missionPlanner.mode === 'launcher' ? 'active' : ''}`}
                onClick={() => missionPlanner.setMode('launcher')}
                title="Place Launcher (L) - Click to place launch platform"
              >
                <span className="btn-icon">&#9651;</span>
                <span className="btn-text">Launcher</span>
              </button>
              <button
                className={`planner-btn zone ${missionPlanner.mode === 'zone' ? 'active' : ''}`}
                onClick={() => missionPlanner.setMode('zone')}
                title="Draw Zone (Z) - Click & drag to draw"
              >
                <span className="btn-icon">&#9634;</span>
                <span className="btn-text">Zone</span>
              </button>
            </div>
          </div>

          {/* Separator */}
          <div className="planner-separator" />

          {/* Entity Counts */}
          <div className="planner-section">
            <div className="planner-section-label">ENTITIES</div>
            <div className="planner-counts">
              <div className="planner-count interceptor">
                <span className="count-icon">&#9650;</span>
                <span className="count-value">{missionPlanner.plannedEntities.filter(e => e.type === 'interceptor').length}</span>
              </div>
              <div className="planner-count target">
                <span className="count-icon">&#9679;</span>
                <span className="count-value">{missionPlanner.plannedEntities.filter(e => e.type === 'target').length}</span>
              </div>
              <div className="planner-count launcher">
                <span className="count-icon">&#9651;</span>
                <span className="count-value">{missionPlanner.plannedEntities.filter(e => e.type === 'launcher').length}</span>
              </div>
              <div className="planner-count zone">
                <span className="count-icon">&#9634;</span>
                <span className="count-value">{missionPlanner.plannedZones.length}</span>
              </div>
            </div>
          </div>

          {/* Separator */}
          <div className="planner-separator" />

          {/* Grid & Snap */}
          <div className="planner-section">
            <div className="planner-section-label">HELPERS</div>
            <div className="planner-toggles">
              <button
                className={`planner-toggle ${missionPlanner.showGrid ? 'active' : ''}`}
                onClick={() => missionPlanner.setShowGrid(!missionPlanner.showGrid)}
                title="Toggle Grid (G)"
              >
                <span className="toggle-icon">&#9638;</span>
                <span className="toggle-label">Grid</span>
              </button>
              <button
                className={`planner-toggle ${missionPlanner.snapToGrid ? 'active' : ''}`}
                onClick={() => missionPlanner.setSnapToGrid(!missionPlanner.snapToGrid)}
                title="Snap to Grid (S)"
              >
                <span className="toggle-icon">&#8982;</span>
                <span className="toggle-label">Snap</span>
              </button>
            </div>
          </div>

          {/* Separator */}
          <div className="planner-separator" />

          {/* Actions */}
          <div className="planner-section">
            <div className="planner-section-label">ACTIONS</div>
            <div className="planner-actions">
              {(missionPlanner.plannedEntities.length > 0 || missionPlanner.plannedZones.length > 0) && (
                <button
                  className="planner-btn clear"
                  onClick={missionPlanner.clearAll}
                  title="Clear All (Shift+Delete)"
                >
                  <span className="btn-icon">&#10006;</span>
                  <span className="btn-text">Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Selected Entity Info */}
          {missionPlanner.selectedEntityId && (
            <>
              <div className="planner-separator" />
              <div className="planner-section selected">
                <div className="planner-section-label">SELECTED</div>
                <div className="planner-selected-info">
                  <span className="selected-id">{missionPlanner.selectedEntityId}</span>
                  <button
                    className="planner-btn delete"
                    onClick={() => {
                      const id = missionPlanner.selectedEntityId;
                      if (!id) return;
                      if (id.startsWith('zone_')) {
                        missionPlanner.removeZone(id);
                      } else {
                        missionPlanner.removeEntity(id);
                      }
                    }}
                    title="Delete (Delete/Backspace)"
                  >
                    <span className="btn-icon">&#128465;</span>
                  </button>
                </div>
              </div>
            </>
          )}

          {/* Keyboard Shortcuts Hint */}
          <div className="planner-shortcuts">
            <span>Del: Delete</span>
            <span>Esc: Deselect</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
