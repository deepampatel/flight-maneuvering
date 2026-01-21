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
  const { connected, state, scenarios, startRun, stopRun } = useSimulation();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Air Dominance Simulation</h1>
        <span className="subtitle">Phase 1: Foundation</span>
      </header>

      <main className="app-main">
        <div className="scene-container">
          <SimulationScene state={state} />
        </div>

        <ControlPanel
          connected={connected}
          state={state}
          scenarios={scenarios}
          onStart={startRun}
          onStop={stopRun}
        />
      </main>
    </div>
  );
}

export default App;
