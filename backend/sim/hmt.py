"""
Human-Machine Teaming Module

This module models operator interaction with autonomous systems:
- Authority levels (full auto, human-on-loop, human-in-loop, manual)
- Action proposal and approval workflows
- Operator workload modeling
- Trust calibration metrics

Background:
- Supervisory control theory for human-automation interaction
- Fitts list for function allocation between human and machine
- Situation awareness modeling for operator state

All features are optional and disabled by default.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import time as time_module


class AuthorityLevel(str, Enum):
    """
    Levels of human involvement in autonomous decisions.

    Based on Sheridan's levels of automation.
    """
    FULL_AUTO = "full_auto"           # AI makes all decisions autonomously
    HUMAN_ON_LOOP = "human_on_loop"   # AI acts, human monitors and can intervene
    HUMAN_IN_LOOP = "human_in_loop"   # Human approves each action before execution
    MANUAL = "manual"                  # Human makes all decisions, AI provides info


class ActionType(str, Enum):
    """Types of actions that may require human approval."""
    ENGAGE = "engage"                     # Begin engagement of target
    MANEUVER = "maneuver"                 # Execute tactical maneuver
    HANDOFF = "handoff"                   # Transfer target to another asset
    ABORT = "abort"                       # Abort current action
    WEAPONS_RELEASE = "weapons_release"   # Fire weapon
    MODE_CHANGE = "mode_change"           # Change operating mode
    FORMATION_CHANGE = "formation_change" # Change formation


class ActionStatus(str, Enum):
    """Status of a proposed action."""
    PENDING = "pending"       # Awaiting decision
    APPROVED = "approved"     # Human approved
    REJECTED = "rejected"     # Human rejected
    EXPIRED = "expired"       # Timed out without decision
    AUTO_APPROVED = "auto_approved"  # Auto-approved by system


@dataclass
class PendingAction:
    """
    An action proposed by the AI awaiting human decision.
    """
    action_id: str
    action_type: ActionType
    entity_id: str                # Entity that will perform action
    target_id: Optional[str]      # Target of action (if applicable)
    proposed_by: str              # "ai" or "operator"
    confidence: float             # AI confidence in this action (0-1)
    details: Dict[str, Any]       # Additional action details
    timestamp: float              # When action was proposed
    timeout: float                # Seconds until auto-timeout
    status: ActionStatus = ActionStatus.PENDING
    decision_time: Optional[float] = None  # When decision was made
    decision_reason: Optional[str] = None  # Why approved/rejected

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "entity_id": self.entity_id,
            "target_id": self.target_id,
            "proposed_by": self.proposed_by,
            "confidence": self.confidence,
            "details": self.details,
            "timestamp": self.timestamp,
            "timeout": self.timeout,
            "status": self.status.value,
            "decision_time": self.decision_time,
            "decision_reason": self.decision_reason,
            "time_remaining": max(0, self.timeout - (time_module.time() - self.timestamp)) if self.status == ActionStatus.PENDING else 0,
        }

    @classmethod
    def create(
        cls,
        action_type: ActionType,
        entity_id: str,
        target_id: Optional[str] = None,
        confidence: float = 0.5,
        details: Optional[Dict[str, Any]] = None,
        timestamp: float = 0.0,
        timeout: float = 5.0
    ) -> "PendingAction":
        """Create a new pending action with auto-generated ID."""
        return cls(
            action_id=str(uuid.uuid4())[:8],
            action_type=action_type,
            entity_id=entity_id,
            target_id=target_id,
            proposed_by="ai",
            confidence=confidence,
            details=details or {},
            timestamp=timestamp,
            timeout=timeout,
        )


@dataclass
class HMTConfig:
    """
    Configuration for human-machine teaming.
    """
    # Authority settings
    authority_level: AuthorityLevel = AuthorityLevel.HUMAN_ON_LOOP
    approval_timeout: float = 5.0  # Seconds to wait for approval

    # Actions requiring explicit approval (regardless of authority level)
    require_approval_types: List[ActionType] = field(default_factory=lambda: [
        ActionType.WEAPONS_RELEASE
    ])

    # Auto-approval settings
    confidence_threshold: float = 0.8   # Auto-approve if AI confidence above this
    auto_approve_on_timeout: bool = True  # In human-on-loop, approve on timeout

    # Workload settings
    max_concurrent_decisions: int = 5   # Max pending actions before degraded mode
    decision_fatigue_threshold: int = 20  # Actions per minute before fatigue

    def to_dict(self) -> dict:
        return {
            "authority_level": self.authority_level.value,
            "approval_timeout": self.approval_timeout,
            "require_approval_types": [t.value for t in self.require_approval_types],
            "confidence_threshold": self.confidence_threshold,
            "auto_approve_on_timeout": self.auto_approve_on_timeout,
            "max_concurrent_decisions": self.max_concurrent_decisions,
            "decision_fatigue_threshold": self.decision_fatigue_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HMTConfig":
        authority = data.get("authority_level", "human_on_loop")
        if isinstance(authority, str):
            authority = AuthorityLevel(authority)

        require_types = data.get("require_approval_types", ["weapons_release"])
        require_types = [ActionType(t) if isinstance(t, str) else t for t in require_types]

        return cls(
            authority_level=authority,
            approval_timeout=data.get("approval_timeout", 5.0),
            require_approval_types=require_types,
            confidence_threshold=data.get("confidence_threshold", 0.8),
            auto_approve_on_timeout=data.get("auto_approve_on_timeout", True),
            max_concurrent_decisions=data.get("max_concurrent_decisions", 5),
            decision_fatigue_threshold=data.get("decision_fatigue_threshold", 20),
        )


@dataclass
class WorkloadMetrics:
    """Operator workload metrics."""
    actions_per_minute: float = 0.0
    pending_actions: int = 0
    response_time_avg_ms: float = 0.0
    missed_deadlines: int = 0
    total_decisions: int = 0
    fatigue_level: float = 0.0  # 0-1, based on decision rate

    def to_dict(self) -> dict:
        return {
            "actions_per_minute": self.actions_per_minute,
            "pending_actions": self.pending_actions,
            "response_time_avg_ms": self.response_time_avg_ms,
            "missed_deadlines": self.missed_deadlines,
            "total_decisions": self.total_decisions,
            "fatigue_level": self.fatigue_level,
        }


@dataclass
class TrustMetrics:
    """Trust calibration between human and AI."""
    ai_accuracy: float = 1.0           # Fraction of AI decisions that were correct
    human_override_rate: float = 0.0   # Fraction of AI decisions overridden by human
    automation_reliance: float = 0.5   # How much human relies on AI (0-1)
    agreement_rate: float = 1.0        # How often human agrees with AI

    # Counters for calculation
    _ai_correct: int = 0
    _ai_total: int = 0
    _human_overrides: int = 0
    _agreements: int = 0

    def to_dict(self) -> dict:
        return {
            "ai_accuracy": self.ai_accuracy,
            "human_override_rate": self.human_override_rate,
            "automation_reliance": self.automation_reliance,
            "agreement_rate": self.agreement_rate,
        }

    def record_outcome(self, ai_recommended: bool, human_approved: bool, outcome_correct: bool) -> None:
        """Record an interaction outcome for trust calibration."""
        self._ai_total += 1

        if outcome_correct:
            self._ai_correct += 1

        if ai_recommended and not human_approved:
            self._human_overrides += 1

        if ai_recommended == human_approved:
            self._agreements += 1

        # Update metrics
        if self._ai_total > 0:
            self.ai_accuracy = self._ai_correct / self._ai_total
            self.human_override_rate = self._human_overrides / self._ai_total
            self.agreement_rate = self._agreements / self._ai_total

        # Automation reliance based on agreement rate
        self.automation_reliance = self.agreement_rate * self.ai_accuracy


class HumanMachineTeaming:
    """
    Models human-machine teaming for supervisory control.

    Features:
    - Authority level management
    - Action proposal and approval workflow
    - Workload tracking
    - Trust calibration

    Usage:
        hmt = HumanMachineTeaming(config)

        # AI proposes an action
        action_id = hmt.propose_action(action)

        # Human approves or rejects
        hmt.approve_action(action_id)
        # or
        hmt.reject_action(action_id, reason="Too risky")

        # Update each tick
        timed_out = hmt.update(current_time)
    """

    def __init__(self, config: Optional[HMTConfig] = None):
        self.config = config or HMTConfig()

        # Action tracking
        self.pending_actions: Dict[str, PendingAction] = {}
        self.action_history: List[PendingAction] = []

        # Metrics
        self.workload = WorkloadMetrics()
        self.trust = TrustMetrics()

        # Response time tracking
        self._response_times: List[float] = []
        self._decision_times: List[float] = []  # Timestamps of recent decisions

    def set_authority_level(self, level: AuthorityLevel) -> None:
        """Change the authority level."""
        self.config.authority_level = level

    def propose_action(self, action: PendingAction) -> str:
        """
        AI proposes an action for approval.

        Returns action_id for tracking.

        Depending on authority level and confidence, may auto-approve.
        """
        self.pending_actions[action.action_id] = action
        self.workload.pending_actions = len(self.pending_actions)

        # Check for auto-approval
        if self._should_auto_approve(action):
            self._approve_internal(action.action_id, auto=True)

        return action.action_id

    def _should_auto_approve(self, action: PendingAction) -> bool:
        """Determine if action should be auto-approved."""
        # Full auto mode: always auto-approve
        if self.config.authority_level == AuthorityLevel.FULL_AUTO:
            return True

        # Manual mode: never auto-approve
        if self.config.authority_level == AuthorityLevel.MANUAL:
            return False

        # Human-in-loop: never auto-approve
        if self.config.authority_level == AuthorityLevel.HUMAN_IN_LOOP:
            return False

        # Critical actions always require approval
        if action.action_type in self.config.require_approval_types:
            return False

        # High confidence actions in human-on-loop mode
        if (self.config.authority_level == AuthorityLevel.HUMAN_ON_LOOP and
            action.confidence >= self.config.confidence_threshold):
            return True

        return False

    def approve_action(self, action_id: str, reason: Optional[str] = None) -> bool:
        """
        Human approves a pending action.

        Returns True if action was found and approved.
        """
        return self._approve_internal(action_id, auto=False, reason=reason)

    def _approve_internal(self, action_id: str, auto: bool = False, reason: Optional[str] = None) -> bool:
        """Internal approval handler."""
        if action_id not in self.pending_actions:
            return False

        action = self.pending_actions.pop(action_id)
        action.status = ActionStatus.AUTO_APPROVED if auto else ActionStatus.APPROVED
        action.decision_time = time_module.time()
        action.decision_reason = reason

        self.action_history.append(action)
        self._record_decision(action, approved=True)

        return True

    def reject_action(
        self,
        action_id: str,
        reason: Optional[str] = None,
        alternative: Optional[Dict] = None
    ) -> bool:
        """
        Human rejects a pending action.

        Optionally provide a reason or alternative action.

        Returns True if action was found and rejected.
        """
        if action_id not in self.pending_actions:
            return False

        action = self.pending_actions.pop(action_id)
        action.status = ActionStatus.REJECTED
        action.decision_time = time_module.time()
        action.decision_reason = reason

        if alternative:
            action.details["alternative"] = alternative

        self.action_history.append(action)
        self._record_decision(action, approved=False)

        # Record override for trust metrics
        self.trust.record_outcome(
            ai_recommended=True,
            human_approved=False,
            outcome_correct=True  # Assume human is correct when overriding
        )

        return True

    def _record_decision(self, action: PendingAction, approved: bool) -> None:
        """Record decision metrics."""
        current_time = time_module.time()

        # Response time
        response_time = (action.decision_time or current_time) - action.timestamp
        self._response_times.append(response_time * 1000)  # to ms
        if len(self._response_times) > 50:
            self._response_times.pop(0)

        # Decision time tracking
        self._decision_times.append(current_time)

        # Update workload metrics
        self.workload.total_decisions += 1
        self.workload.pending_actions = len(self.pending_actions)

        if self._response_times:
            self.workload.response_time_avg_ms = sum(self._response_times) / len(self._response_times)

    def update(self, current_time: float) -> List[PendingAction]:
        """
        Process timeouts and update metrics.

        Call once per simulation tick.

        Returns:
            List of actions that timed out
        """
        timed_out = []

        # Check for timeouts
        for action_id, action in list(self.pending_actions.items()):
            elapsed = current_time - action.timestamp

            if elapsed > action.timeout:
                # Handle timeout based on authority level
                if (self.config.authority_level == AuthorityLevel.HUMAN_ON_LOOP and
                    self.config.auto_approve_on_timeout and
                    action.action_type not in self.config.require_approval_types):
                    # Auto-approve on timeout in human-on-loop mode
                    self._approve_internal(action_id, auto=True, reason="timeout_auto_approve")
                else:
                    # Expire the action
                    action.status = ActionStatus.EXPIRED
                    action.decision_time = current_time
                    self.action_history.append(action)
                    del self.pending_actions[action_id]
                    timed_out.append(action)
                    self.workload.missed_deadlines += 1

        # Update workload metrics
        self._update_workload_metrics(current_time)

        return timed_out

    def _update_workload_metrics(self, current_time: float) -> None:
        """Update workload and fatigue metrics."""
        # Calculate actions per minute
        one_minute_ago = current_time - 60

        recent_decisions = [t for t in self._decision_times if t > one_minute_ago]
        self.workload.actions_per_minute = len(recent_decisions)

        # Calculate fatigue level based on decision rate
        fatigue_threshold = self.config.decision_fatigue_threshold
        if fatigue_threshold > 0:
            self.workload.fatigue_level = min(1.0, self.workload.actions_per_minute / fatigue_threshold)
        else:
            self.workload.fatigue_level = 0.0

        # Update pending count
        self.workload.pending_actions = len(self.pending_actions)

    def get_pending_actions(self) -> List[PendingAction]:
        """Get list of actions awaiting human decision."""
        return list(self.pending_actions.values())

    def get_action_history(self, limit: int = 20) -> List[PendingAction]:
        """Get recent action history."""
        return self.action_history[-limit:]

    def get_recommended_action(
        self,
        action_type: ActionType,
        entity_id: str,
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        Get AI recommendation for which candidate to select.

        Useful for target selection, formation choice, etc.
        """
        if not candidates:
            return None

        # Simple heuristic: return highest scoring candidate
        # In real system, would use ML model
        if candidates:
            # Assume candidates have a "score" field
            scored = [c for c in candidates if "score" in c]
            if scored:
                return max(scored, key=lambda c: c["score"])
            return candidates[0]

        return None

    def is_action_allowed(self, action_type: ActionType, confidence: float = 0.5) -> bool:
        """
        Check if an action type would be auto-approved.

        Useful for AI to know whether to proceed without waiting.
        """
        if self.config.authority_level == AuthorityLevel.FULL_AUTO:
            return True

        if self.config.authority_level == AuthorityLevel.MANUAL:
            return False

        if action_type in self.config.require_approval_types:
            return False

        if self.config.authority_level == AuthorityLevel.HUMAN_ON_LOOP:
            return confidence >= self.config.confidence_threshold

        return False

    def record_action_outcome(
        self,
        action_id: str,
        success: bool,
        details: Optional[Dict] = None
    ) -> None:
        """
        Record the outcome of an executed action.

        Used for trust calibration.
        """
        # Find action in history
        action = next((a for a in self.action_history if a.action_id == action_id), None)
        if action is None:
            return

        # Record for trust metrics
        ai_recommended = action.proposed_by == "ai"
        human_approved = action.status in [ActionStatus.APPROVED, ActionStatus.AUTO_APPROVED]

        self.trust.record_outcome(
            ai_recommended=ai_recommended,
            human_approved=human_approved,
            outcome_correct=success
        )

    def get_metrics(self) -> Dict:
        """Get all HMT metrics."""
        return {
            "workload": self.workload.to_dict(),
            "trust": self.trust.to_dict(),
            "authority_level": self.config.authority_level.value,
            "pending_count": len(self.pending_actions),
        }


def get_authority_levels() -> List[dict]:
    """Get list of available authority levels with descriptions."""
    return [
        {
            "id": "full_auto",
            "name": "Full Auto",
            "description": "AI makes all decisions autonomously"
        },
        {
            "id": "human_on_loop",
            "name": "Human-on-Loop",
            "description": "AI acts, human monitors and can intervene"
        },
        {
            "id": "human_in_loop",
            "name": "Human-in-Loop",
            "description": "Human approves each action before execution"
        },
        {
            "id": "manual",
            "name": "Manual",
            "description": "Human makes all decisions, AI provides info"
        },
    ]
