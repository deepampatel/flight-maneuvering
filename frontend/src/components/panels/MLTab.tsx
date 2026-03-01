/**
 * MLTab — ML/AI model status
 *
 * Shows ONNX Runtime availability and loaded threat/guidance models.
 */

import type { MLStatus } from '../../types';

interface MLTabProps {
  mlStatus?: MLStatus | null;
}

export function MLTab({ mlStatus }: MLTabProps) {
  return (
    <div className="advanced-content">
      <p className="panel-desc">Neural network threat assessment and RL guidance policies</p>

      <div className="ml-status">
        <div className="ml-status-row">
          <span className="hud-label">ONNX Runtime</span>
          <span className={`hud-value ${mlStatus?.onnx_available ? 'result-intercept' : 'result-missed'}`}>
            {mlStatus?.onnx_available ? 'AVAILABLE' : 'NOT INSTALLED'}
          </span>
        </div>

        {!mlStatus?.onnx_available && (
          <div className="ml-install-hint">
            <code>pip install onnxruntime</code>
          </div>
        )}

        <div className="ml-section">
          <div className="ml-section-title">Threat Models</div>
          {mlStatus?.models?.threat_models && mlStatus.models.threat_models.length > 0 ? (
            mlStatus.models.threat_models.map((model) => (
              <div key={model.model_id} className="ml-model-row">
                <span className="model-id">{model.model_id}</span>
                <span className={`model-status ${model.active ? 'active' : ''}`}>
                  {model.active ? 'ACTIVE' : model.loaded ? 'LOADED' : 'UNLOADED'}
                </span>
              </div>
            ))
          ) : (
            <div className="ml-no-models">No models loaded</div>
          )}
        </div>

        <div className="ml-section">
          <div className="ml-section-title">Guidance Models</div>
          {mlStatus?.models?.guidance_models && mlStatus.models.guidance_models.length > 0 ? (
            mlStatus.models.guidance_models.map((model) => (
              <div key={model.model_id} className="ml-model-row">
                <span className="model-id">{model.model_id}</span>
                <span className={`model-status ${model.active ? 'active' : ''}`}>
                  {model.active ? 'ACTIVE' : model.loaded ? 'LOADED' : 'UNLOADED'}
                </span>
              </div>
            ))
          ) : (
            <div className="ml-no-models">No models loaded</div>
          )}
        </div>

        <div className="ml-info">
          <p>Load ONNX models via API:</p>
          <code>POST /ml/models/load</code>
        </div>
      </div>
    </div>
  );
}
