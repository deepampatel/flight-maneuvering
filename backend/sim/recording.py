"""
Recording & Replay - Engagement Capture and Playback

This module provides the ability to:
1. Record complete engagement history (all entity states per tick)
2. Save recordings to disk for later analysis
3. Replay recordings at variable speed
4. Enable "what if" analysis by replaying with different parameters

ARCHITECTURE:

RecordingManager: Handles recording lifecycle
- Start/stop recording
- Save/load to disk
- Manage recording metadata

ReplayEngine: Plays back recorded engagements
- Variable speed playback (0.5x to 4x)
- Seek to specific tick
- Emit events via WebSocket (same as live sim)

FILE FORMAT:
JSON files stored in ./recordings/ directory
Named by recording_id (UUID)
Contains all frames plus metadata
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Awaitable, Any
import json
import os
import time
import uuid
import asyncio
from pathlib import Path

from .vector import Vec3
from .entities import Entity


@dataclass
class RecordedFrame:
    """
    Single frame of recorded simulation data.

    Contains complete state at one tick - enough to reconstruct
    the simulation at any point.
    """
    tick: int
    sim_time: float

    # Entity states (serialized to dict)
    target_state: Dict[str, Any]
    interceptor_states: List[Dict[str, Any]]

    # Optional computed data at this frame
    intercept_geometries: Optional[List[Dict[str, Any]]] = None
    threat_assessments: Optional[List[Dict[str, Any]]] = None

    # Guidance commands issued this tick
    guidance_commands: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize frame for storage."""
        return {
            "tick": self.tick,
            "sim_time": self.sim_time,
            "target_state": self.target_state,
            "interceptor_states": self.interceptor_states,
            "intercept_geometries": self.intercept_geometries,
            "threat_assessments": self.threat_assessments,
            "guidance_commands": self.guidance_commands
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedFrame":
        """Deserialize frame from storage."""
        return cls(
            tick=data["tick"],
            sim_time=data["sim_time"],
            target_state=data["target_state"],
            interceptor_states=data["interceptor_states"],
            intercept_geometries=data.get("intercept_geometries"),
            threat_assessments=data.get("threat_assessments"),
            guidance_commands=data.get("guidance_commands")
        )


@dataclass
class EngagementRecording:
    """
    Complete recording of an engagement.

    Contains all metadata plus every frame of the simulation.
    """
    # Metadata
    recording_id: str
    created_at: float
    scenario_name: str

    # Configuration used
    config: Dict[str, Any]

    # All frames (one per tick)
    frames: List[RecordedFrame] = field(default_factory=list)

    # Outcome
    result: str = "pending"  # 'intercept', 'missed', 'timeout'
    final_miss_distance: float = 0.0
    total_sim_time: float = 0.0
    total_ticks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "recording_id": self.recording_id,
            "created_at": self.created_at,
            "scenario_name": self.scenario_name,
            "config": self.config,
            "frames": [f.to_dict() for f in self.frames],
            "result": self.result,
            "final_miss_distance": self.final_miss_distance,
            "total_sim_time": self.total_sim_time,
            "total_ticks": self.total_ticks
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Get metadata without frames (for listing)."""
        return {
            "recording_id": self.recording_id,
            "created_at": self.created_at,
            "scenario_name": self.scenario_name,
            "result": self.result,
            "final_miss_distance": round(self.final_miss_distance, 1),
            "total_sim_time": round(self.total_sim_time, 2),
            "total_ticks": self.total_ticks,
            "guidance": self.config.get("guidance_type", "unknown"),
            "evasion": self.config.get("evasion_type", "none")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngagementRecording":
        """Deserialize from storage."""
        recording = cls(
            recording_id=data["recording_id"],
            created_at=data["created_at"],
            scenario_name=data["scenario_name"],
            config=data["config"],
            result=data.get("result", "pending"),
            final_miss_distance=data.get("final_miss_distance", 0.0),
            total_sim_time=data.get("total_sim_time", 0.0),
            total_ticks=data.get("total_ticks", 0)
        )
        recording.frames = [RecordedFrame.from_dict(f) for f in data.get("frames", [])]
        return recording


@dataclass
class ReplayConfig:
    """Configuration for replay playback."""
    speed_multiplier: float = 1.0  # 0.5 = half speed, 2.0 = double
    start_tick: int = 0
    end_tick: Optional[int] = None


class RecordingManager:
    """
    Manages recording and replay of engagements.

    Usage:
        manager = RecordingManager()
        manager.start_recording("head_on", config_dict)
        # During simulation:
        manager.record_frame(sim_state)
        # When done:
        recording = manager.stop_recording("intercept", 10.5)
        manager.save_recording(recording)
    """

    def __init__(self, storage_dir: str = "./recordings"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.active_recording: Optional[EngagementRecording] = None
        self.is_recording: bool = False

    def start_recording(self, scenario_name: str, config: Dict[str, Any]) -> str:
        """
        Start recording a new engagement.

        Args:
            scenario_name: Name of the scenario being run
            config: Configuration dict (guidance, evasion, etc.)

        Returns:
            recording_id for the new recording
        """
        recording_id = str(uuid.uuid4())[:8]

        self.active_recording = EngagementRecording(
            recording_id=recording_id,
            created_at=time.time(),
            scenario_name=scenario_name,
            config=config
        )
        self.is_recording = True

        return recording_id

    def record_frame(
        self,
        tick: int,
        sim_time: float,
        target: Entity,
        interceptors: List[Entity],
        intercept_geometries: Optional[List[Dict]] = None,
        threat_assessments: Optional[List[Dict]] = None,
        guidance_commands: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Record a single frame during simulation.

        Called once per tick to capture the complete state.
        """
        if not self.is_recording or self.active_recording is None:
            return

        frame = RecordedFrame(
            tick=tick,
            sim_time=sim_time,
            target_state=target.to_state_dict(),
            interceptor_states=[i.to_state_dict() for i in interceptors],
            intercept_geometries=intercept_geometries,
            threat_assessments=threat_assessments,
            guidance_commands=guidance_commands
        )

        self.active_recording.frames.append(frame)

    def stop_recording(
        self,
        result: str,
        miss_distance: float,
        sim_time: float
    ) -> Optional[EngagementRecording]:
        """
        Finalize and return the recording.

        Args:
            result: Engagement result ('intercept', 'missed', 'timeout')
            miss_distance: Final miss distance in meters
            sim_time: Total simulation time

        Returns:
            The completed EngagementRecording
        """
        if not self.is_recording or self.active_recording is None:
            return None

        self.active_recording.result = result
        self.active_recording.final_miss_distance = miss_distance
        self.active_recording.total_sim_time = sim_time
        self.active_recording.total_ticks = len(self.active_recording.frames)

        recording = self.active_recording
        self.active_recording = None
        self.is_recording = False

        return recording

    def save_recording(self, recording: EngagementRecording) -> str:
        """
        Save recording to disk.

        Returns:
            Filepath where recording was saved
        """
        filepath = self.storage_dir / f"{recording.recording_id}.json"

        with open(filepath, 'w') as f:
            json.dump(recording.to_dict(), f)

        return str(filepath)

    def load_recording(self, recording_id: str) -> Optional[EngagementRecording]:
        """
        Load recording from disk.

        Args:
            recording_id: ID of the recording to load

        Returns:
            EngagementRecording or None if not found
        """
        filepath = self.storage_dir / f"{recording_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        return EngagementRecording.from_dict(data)

    def list_recordings(self) -> List[Dict[str, Any]]:
        """
        List all saved recordings with metadata.

        Returns:
            List of recording metadata dicts
        """
        recordings = []

        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Create recording to get metadata
                recording = EngagementRecording.from_dict(data)
                recordings.append(recording.to_metadata())
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted files
                continue

        # Sort by created_at, newest first
        recordings.sort(key=lambda r: r["created_at"], reverse=True)

        return recordings

    def delete_recording(self, recording_id: str) -> bool:
        """
        Delete a recording.

        Returns:
            True if deleted, False if not found
        """
        filepath = self.storage_dir / f"{recording_id}.json"

        if filepath.exists():
            filepath.unlink()
            return True

        return False


class ReplayEngine:
    """
    Plays back recorded engagements.

    Emits events via registered handlers (same format as live simulation).
    Supports variable speed playback and seeking.

    Usage:
        engine = ReplayEngine(recording)
        engine.on_event(my_handler)
        await engine.play()
    """

    def __init__(
        self,
        recording: EngagementRecording,
        config: Optional[ReplayConfig] = None
    ):
        self.recording = recording
        self.config = config or ReplayConfig()

        self.current_tick: int = self.config.start_tick
        self.is_playing: bool = False
        self.is_paused: bool = False

        self._event_handlers: List[Callable[[Dict], Awaitable[None]]] = []
        self._play_task: Optional[asyncio.Task] = None

    def on_event(self, handler: Callable[[Dict], Awaitable[None]]) -> None:
        """Register event handler for replay events."""
        self._event_handlers.append(handler)

    async def _emit_event(self, event: Dict) -> None:
        """Emit event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception:
                pass  # Don't let handler errors stop replay

    def _frame_to_event(self, frame: RecordedFrame) -> Dict:
        """Convert recorded frame to event format matching live sim."""
        # Calculate miss distance from entity positions
        miss_distance = float('inf')
        if frame.target_state and frame.interceptor_states:
            target_pos = frame.target_state.get("position", {})
            for interceptor_state in frame.interceptor_states:
                int_pos = interceptor_state.get("position", {})
                if target_pos and int_pos:
                    dx = target_pos.get("x", 0) - int_pos.get("x", 0)
                    dy = target_pos.get("y", 0) - int_pos.get("y", 0)
                    dz = target_pos.get("z", 0) - int_pos.get("z", 0)
                    dist = (dx**2 + dy**2 + dz**2) ** 0.5
                    miss_distance = min(miss_distance, dist)

        return {
            "type": "state",
            "run_id": self.recording.recording_id,
            "ts": time.time(),
            "sim_time": frame.sim_time,
            "tick": frame.tick,
            "status": "running",
            "result": "pending",
            "entities": [frame.target_state] + frame.interceptor_states,
            "miss_distance": miss_distance if miss_distance != float('inf') else 0,
            "replay": True,  # Flag to indicate this is replay
            "intercept_geometries": frame.intercept_geometries,
            "threat_assessments": frame.threat_assessments
        }

    async def play(self) -> None:
        """
        Play the recording, emitting events.

        Respects speed_multiplier from config.
        """
        if self.is_playing:
            return

        self.is_playing = True
        self.is_paused = False

        # Determine tick range
        end_tick = self.config.end_tick or len(self.recording.frames)
        end_tick = min(end_tick, len(self.recording.frames))

        # Calculate base delay (assuming 50 Hz simulation)
        base_dt = 0.02
        actual_dt = base_dt / self.config.speed_multiplier

        while self.current_tick < end_tick and self.is_playing:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            frame = self.recording.frames[self.current_tick]
            event = self._frame_to_event(frame)
            await self._emit_event(event)

            self.current_tick += 1
            await asyncio.sleep(actual_dt)

        # Emit completion event
        if self.is_playing:
            await self._emit_event({
                "type": "complete",
                "run_id": self.recording.recording_id,
                "result": self.recording.result,
                "miss_distance": self.recording.final_miss_distance,
                "sim_time": self.recording.total_sim_time,
                "replay": True
            })

        self.is_playing = False

    async def pause(self) -> None:
        """Pause playback."""
        self.is_paused = True

    async def resume(self) -> None:
        """Resume playback after pause."""
        self.is_paused = False

    async def seek(self, tick: int) -> None:
        """
        Jump to specific tick.

        Emits the frame at that tick immediately.
        """
        tick = max(0, min(tick, len(self.recording.frames) - 1))
        self.current_tick = tick

        if tick < len(self.recording.frames):
            frame = self.recording.frames[tick]
            event = self._frame_to_event(frame)
            await self._emit_event(event)

    async def stop(self) -> None:
        """Stop playback."""
        self.is_playing = False
        self.is_paused = False

    def get_frame(self, tick: int) -> Optional[RecordedFrame]:
        """Get specific frame without playing."""
        if 0 <= tick < len(self.recording.frames):
            return self.recording.frames[tick]
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current replay state."""
        return {
            "recording_id": self.recording.recording_id,
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "current_tick": self.current_tick,
            "total_ticks": len(self.recording.frames),
            "speed_multiplier": self.config.speed_multiplier,
            "scenario_name": self.recording.scenario_name,
            "result": self.recording.result
        }


# Global recording manager instance
_recording_manager: Optional[RecordingManager] = None


def get_recording_manager() -> RecordingManager:
    """Get or create the global recording manager."""
    global _recording_manager
    if _recording_manager is None:
        _recording_manager = RecordingManager()
    return _recording_manager
