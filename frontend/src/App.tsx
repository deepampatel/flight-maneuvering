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
import { useSimulation } from './hooks/useSimulation';
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
  } = useSimulation();

  const [showAdvanced, setShowAdvanced] = useState(false);

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
            onStart={startRun}
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
          />
        </div>
      </header>

      <main className="app-main">
        <div className="scene-container">
          <SimulationScene state={state} interceptGeometry={interceptGeometry} />
        </div>
      </main>
    </div>
  );
}

export default App;
