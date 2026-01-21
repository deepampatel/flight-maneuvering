/**
 * Main App Component
 *
 * Layout:
 * - Left: 3D visualization (takes most space)
 * - Right: Control panel with telemetry
 */

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
  } = useSimulation();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Air Dominance Simulation</h1>
        <span className="subtitle">Phase 3: Evasion + Multi-Interceptor + Envelope</span>
      </header>

      <main className="app-main">
        <div className="scene-container">
          <SimulationScene state={state} />
        </div>

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
        />
      </main>
    </div>
  );
}

export default App;
