/**
 * AnalysisTab — Monte Carlo + Engagement Envelope
 *
 * Runs statistical analysis: 100-run Monte Carlo for Pk estimation
 * and range/bearing intercept envelope heatmap.
 */

import type { MonteCarloResults, EnvelopeResults } from '../../types';

interface AnalysisTabProps {
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
  mcResults: MonteCarloResults | null;
  envelopeResults: EnvelopeResults | null;
  onMcResults: (r: MonteCarloResults) => void;
  onEnvelopeResults: (r: EnvelopeResults) => void;
}

export function AnalysisTab({
  selectedScenario,
  selectedGuidance,
  navConstant,
  selectedEvasion,
  onRunMonteCarlo,
  onRunEnvelope,
  monteCarloLoading,
  envelopeLoading,
  mcResults,
  envelopeResults,
  onMcResults,
  onEnvelopeResults,
}: AnalysisTabProps) {
  const handleRunMonteCarlo = async () => {
    const results = await onRunMonteCarlo({
      scenario: selectedScenario,
      guidance: selectedGuidance,
      navConstant,
      numRuns: 100,
      killRadius: 50,
      positionNoiseStd: 50,
      velocityNoiseStd: 5,
    });
    onMcResults(results);
  };

  const handleRunEnvelope = async () => {
    const results = await onRunEnvelope({
      guidance: selectedGuidance,
      nav_constant: navConstant,
      evasion: selectedEvasion,
      range_steps: 8,
      bearing_steps: 10,
      runs_per_point: 5,
    });
    onEnvelopeResults(results);
  };

  return (
    <div className="advanced-content">
      {/* Monte Carlo */}
      <div className="analysis-section">
        <h4 className="section-title">MONTE CARLO ANALYSIS</h4>
        <p className="panel-desc">Run 100 simulations with noise to test robustness</p>
        <button
          onClick={handleRunMonteCarlo}
          disabled={monteCarloLoading}
          className="btn-action"
        >
          {monteCarloLoading ? 'Running...' : 'Run Monte Carlo'}
        </button>

        {mcResults && (
          <div className="mc-results">
            <div className="results-grid">
              <div className="result-item">
                <span className="result-label">Pk</span>
                <span className={mcResults.intercept_rate > 0.8 ? 'result-good' : 'result-bad'}>
                  {(mcResults.intercept_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="result-item">
                <span className="result-label">Mean</span>
                <span>{mcResults.mean_miss_distance.toFixed(1)}m</span>
              </div>
              <div className="result-item">
                <span className="result-label">StdDev</span>
                <span>{mcResults.std_miss_distance.toFixed(1)}m</span>
              </div>
              <div className="result-item">
                <span className="result-label">Range</span>
                <span>{mcResults.min_miss_distance.toFixed(0)}-{mcResults.max_miss_distance.toFixed(0)}m</span>
              </div>
            </div>
            <div className="histogram">
              {mcResults.miss_distance_histogram.counts.map((count, i) => {
                const maxCount = Math.max(...mcResults.miss_distance_histogram.counts);
                const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                return (
                  <div
                    key={i}
                    className="hist-bar"
                    style={{ height: `${height}%` }}
                    title={`${mcResults.miss_distance_histogram.bin_edges[i].toFixed(0)}-${mcResults.miss_distance_histogram.bin_edges[i + 1].toFixed(0)}m: ${count}`}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Envelope */}
      <div className="analysis-section" style={{ marginTop: '16px' }}>
        <h4 className="section-title">ENGAGEMENT ENVELOPE</h4>
        <p className="panel-desc">Compute intercept probability across range and bearing</p>
        <button
          onClick={handleRunEnvelope}
          disabled={envelopeLoading}
          className="btn-action"
        >
          {envelopeLoading ? 'Computing...' : 'Compute Envelope'}
        </button>

        {envelopeResults && (
          <div className="envelope-results">
            <div className="heatmap">
              {envelopeResults.heatmap_2d.data.map((row, ri) => (
                <div key={ri} className="heatmap-row">
                  {row.map((value, ci) => {
                    const hue = value * 120;
                    return (
                      <div
                        key={ci}
                        className="heatmap-cell"
                        style={{ backgroundColor: `hsl(${hue}, 80%, 40%)` }}
                        title={`R:${envelopeResults.range_values[ri].toFixed(0)}m B:${envelopeResults.bearing_values[ci].toFixed(0)}\u00b0 Pk:${(value * 100).toFixed(0)}%`}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="heatmap-legend">
              <span>0%</span>
              <div className="gradient-bar" />
              <span>100%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
