/**
 * RecordingsTab — Recording list + replay launch
 *
 * Shows saved simulation recordings with play/delete controls.
 */

import type { RecordingMetadata } from '../../types';

interface RecordingsTabProps {
  recordings: RecordingMetadata[];
  onStartReplay: (id: string) => void;
  onDeleteRecording: (id: string) => void;
}

export function RecordingsTab({
  recordings,
  onStartReplay,
  onDeleteRecording,
}: RecordingsTabProps) {
  return (
    <div className="advanced-content">
      {recordings.length > 0 ? (
        <div className="recordings-list">
          {recordings.slice(0, 8).map((rec) => (
            <div key={rec.recording_id} className="recording-row">
              <div className="rec-info">
                <span className="rec-name">{rec.scenario_name}</span>
                <span className={`rec-result result-${rec.result}`}>{rec.result}</span>
                <span className="rec-time">{rec.total_sim_time.toFixed(1)}s</span>
              </div>
              <div className="rec-actions">
                <button onClick={() => onStartReplay(rec.recording_id)}>PLAY</button>
                <button onClick={() => onDeleteRecording(rec.recording_id)}>DEL</button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="panel-desc">No recordings. Start recording during simulation.</p>
      )}
    </div>
  );
}
