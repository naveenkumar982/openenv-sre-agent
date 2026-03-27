"""
Cloud SRE OpenEnv Environment.
Implements the full OpenEnv spec: step(), reset(), state().
"""

import copy
from typing import Tuple, Dict, Any, List, Optional
from models import (
    Observation, Action, StepResult, ActionCommand,
    Resource, ResourceStatus, ResourceType
)
from tasks import get_task, Task2LatencySpikeRemediation  # type: ignore


class CloudSREEnv:
    """
    An OpenEnv-compliant environment simulating Cloud SRE operations.

    The agent manages a simulated cloud infrastructure: diagnosing outages,
    terminating idle resources, scaling services, and optimizing costs.
    """

    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self._state: Dict[str, Any] = {}
        self._initial_state: Dict[str, Any] = {}
        self._action_history: List[Dict[str, Any]] = []
        self._current_step: int = 0
        self._done: bool = False
        self._task_id: Optional[str] = None
        self._task_class = None
        self._cumulative_reward: float = 0.0

    # ─── OpenEnv API ──────────────────────────────────────────────────────

    def reset(self, task_id: str = "phantom_volume_cleanup") -> Observation:
        """
        Reset the environment to a specific task's initial state.

        Args:
            task_id: One of 'phantom_volume_cleanup', 'latency_spike_remediation',
                     'noisy_neighbor_incident'.

        Returns:
            Initial Observation for the task.
        """
        self._task_class = get_task(task_id)
        self._task_id = task_id
        self._initial_state = self._task_class.get_initial_state()
        self._state = copy.deepcopy(self._initial_state)
        self._action_history = []
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        return self.state()

    def state(self) -> Observation:
        """Returns the current observation of the environment."""
        return Observation(
            resources=[Resource(**r) for r in self._state.get("resources", [])],
            alerts=[
                # Rebuild Alert from dict
                self._build_alert(a) for a in self._state.get("alerts", [])
            ],
            total_hourly_cost=self._state.get("total_hourly_cost", 0.0),
            system_uptime=self._state.get("system_uptime", 100.0),
            step_number=self._current_step,
            max_steps=self.max_steps,
            budget_limit=self._state.get("budget_limit"),
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute a single action in the environment.

        Args:
            action: The Action to execute.

        Returns:
            StepResult containing (observation, reward, done, info).
        """
        if self._done:
            return StepResult(
                observation=self.state(),
                reward=0.0,
                done=True,
                info={"error": "Episode already finished. Call reset()."}
            )

        self._current_step += 1
        step_reward = 0.0
        info: Dict[str, Any] = {"step": self._current_step}

        # ── Dispatch action ──
        if action.command == ActionCommand.TERMINATE:
            step_reward, msg = self._handle_terminate(action.resource_id)
            info["action_result"] = msg

        elif action.command == ActionCommand.SCALE:
            step_reward, msg = self._handle_scale(
                action.resource_id, action.params.get("target_size", "")
            )
            info["action_result"] = msg

        elif action.command == ActionCommand.REBOOT:
            step_reward, msg = self._handle_reboot(action.resource_id)
            info["action_result"] = msg

        elif action.command == ActionCommand.INSPECT:
            step_reward, msg = self._handle_inspect(action.resource_id)
            info["action_result"] = msg

        elif action.command == ActionCommand.WAIT:
            step_reward = -0.01  # Small penalty for inaction
            info["action_result"] = "Waited one step. No changes made."

        else:
            info["action_result"] = f"Unknown command: {action.command}"

        # ── Record action in history ──
        self._action_history.append({
            "step": self._current_step,
            "command": action.command.value if isinstance(action.command, ActionCommand) else action.command,
            "resource_id": action.resource_id,
            "params": action.params,
        })

        # ── Recalculate derived state ──
        self._recalculate_state()

        # ── Check episode termination ──
        if self._current_step >= self.max_steps:
            self._done = True
            info["termination_reason"] = "max_steps_reached"

        # ── Run grader at end of episode ──
        if self._done:
            final_score, grading = self._task_class.grade(
                self._action_history, self._state, self._initial_state
            )
            info["final_score"] = final_score
            info["grading_breakdown"] = grading

        self._cumulative_reward += step_reward
        info["cumulative_reward"] = round(self._cumulative_reward, 4)

        return StepResult(
            observation=self.state(),
            reward=round(step_reward, 4),
            done=self._done,
            info=info,
        )

    def grade(self) -> Tuple[float, Dict]:
        """
        Run the grader on the current episode.
        Can be called explicitly even before the episode ends.
        """
        if self._task_class is None:
            return 0.0, {"error": "No task loaded. Call reset() first."}
        return self._task_class.grade(
            self._action_history, self._state, self._initial_state
        )

    # ─── Action Handlers ──────────────────────────────────────────────────

    def _handle_terminate(self, resource_id: Optional[str]) -> Tuple[float, str]:
        """Terminate a resource. Returns (step_reward, message)."""
        if not resource_id:
            return -0.05, "Error: No resource_id provided for terminate."

        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        # Dense reward signal: terminating cheap/idle resources is slightly positive,
        # terminating expensive prod resources is very negative
        tags = resource.get("tags", {})
        cost = resource.get("cost_per_hour", 0)
        is_prod = tags.get("env") == "prod"
        is_attached_ebs = (
            resource.get("type") == ResourceType.EBS.value
            and resource.get("status") == ResourceStatus.IN_USE.value
        )

        reward = 0.0
        if is_prod and resource.get("type") == ResourceType.EC2.value:
            reward = -0.15  # Strong negative for killing prod EC2
        elif is_attached_ebs:
            reward = -0.10  # Negative for detaching in-use storage
        elif resource.get("status") == ResourceStatus.AVAILABLE.value:
            reward = 0.05 + (cost * 0.02)  # Positive: cleaning up idle resources
        else:
            reward = 0.02  # Neutral-to-positive for non-prod terminations

        # Remove the resource from state
        self._state["resources"] = [
            r for r in self._state["resources"] if r["id"] != resource_id
        ]

        # Resolve any alerts referencing this resource
        self._resolve_alerts_for(resource_id)

        return reward, f"Terminated resource '{resource_id}'. Cost saved: ${cost:.4f}/hr."

    def _handle_scale(self, resource_id: Optional[str], target_size: str) -> Tuple[float, str]:
        """Scale a resource to a new size. Returns (step_reward, message)."""
        if not resource_id:
            return -0.05, "Error: No resource_id provided for scale."
        if not target_size:
            return -0.05, "Error: No target_size in params for scale."

        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        old_size = resource.get("instance_size", "unknown")
        old_cost = resource.get("cost_per_hour", 0)

        # For RDS, look up new cost from pricing table
        if resource.get("type") == ResourceType.RDS.value:
            pricing = Task2LatencySpikeRemediation.RDS_PRICING
            if target_size not in pricing:
                return -0.05, f"Error: Invalid RDS size '{target_size}'. Valid: {list(pricing.keys())}"
            new_cost = pricing[target_size]
        else:
            # Generic scaling: estimate new cost
            new_cost = old_cost * 2.0  # Simple doubling estimate

        # Update the resource in state
        for r in self._state["resources"]:
            if r["id"] == resource_id:
                r["instance_size"] = target_size
                r["cost_per_hour"] = new_cost
                # Scaling fixes CPU overload
                if r.get("cpu_utilization", 0) > 80:
                    r["cpu_utilization"] = 45.0
                    r["memory_utilization"] = min(r.get("memory_utilization", 50), 60.0)
                break

        # If scaling fixed the problem, resolve related alerts
        self._resolve_alerts_for(resource_id)

        # Positive reward for correct scaling
        reward = 0.08
        # Bonus if we're fixing a high-CPU resource
        if resource.get("cpu_utilization", 0) > 80:
            reward += 0.05

        return reward, (
            f"Scaled '{resource_id}' from {old_size} to {target_size}. "
            f"Cost changed: ${old_cost:.4f}/hr → ${new_cost:.4f}/hr."
        )

    def _handle_reboot(self, resource_id: Optional[str]) -> Tuple[float, str]:
        """Reboot a resource. Returns (step_reward, message)."""
        if not resource_id:
            return -0.05, "Error: No resource_id provided for reboot."

        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        reward = 0.0

        for r in self._state["resources"]:
            if r["id"] == resource_id:
                if r["status"] == ResourceStatus.STOPPED.value:
                    # Rebooting a stopped instance brings it back online
                    r["status"] = ResourceStatus.RUNNING.value
                    r["cpu_utilization"] = 15.0
                    r["memory_utilization"] = 20.0
                    reward = 0.10

                    # Update system uptime since a service is restored
                    self._state["system_uptime"] = min(
                        100.0, self._state.get("system_uptime", 0) + 30.0
                    )

                    # Resolve alerts for this resource
                    self._resolve_alerts_for(resource_id)

                    return reward, f"Rebooted '{resource_id}'. Instance is now RUNNING."
                elif r["status"] == ResourceStatus.RUNNING.value:
                    # Rebooting a running instance — temporary disruption
                    r["status"] = ResourceStatus.REBOOTING.value
                    reward = -0.02
                    # Simulate it coming back next step
                    r["status"] = ResourceStatus.RUNNING.value
                    r["cpu_utilization"] = 10.0
                    return reward, f"Rebooted '{resource_id}'. Temporary disruption."
                else:
                    return -0.05, f"Cannot reboot '{resource_id}' in state '{r['status']}'."

        return -0.05, f"Error: Resource '{resource_id}' not found."

    def _handle_inspect(self, resource_id: Optional[str]) -> Tuple[float, str]:
        """Inspect a resource for detailed info. Returns (step_reward, message)."""
        if not resource_id:
            return -0.01, "Error: No resource_id provided for inspect."

        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.01, f"Error: Resource '{resource_id}' not found."

        # Small positive reward to encourage investigation before action
        reward = 0.01

        # Build a detailed report
        tags = resource.get("tags", {})
        report_lines = [
            f"=== Inspection Report: {resource_id} ===",
            f"  Name:        {resource.get('name', 'N/A')}",
            f"  Type:        {resource.get('type', 'N/A')}",
            f"  Status:      {resource.get('status', 'N/A')}",
            f"  Size:        {resource.get('instance_size', 'N/A')}",
            f"  CPU:         {resource.get('cpu_utilization', 0):.1f}%",
            f"  Memory:      {resource.get('memory_utilization', 0):.1f}%",
            f"  Cost:        ${resource.get('cost_per_hour', 0):.4f}/hr",
            f"  Attached To: {resource.get('attached_to', 'None')}",
            f"  Tags:        {tags}",
        ]

        # Add diagnostic hints based on metrics
        cpu = resource.get("cpu_utilization", 0)
        if cpu >= 95:
            report_lines.append("  ⚠ ALERT: CPU utilization critically high!")
        if tags.get("env") == "test" and cpu > 80:
            report_lines.append("  ⚠ WARNING: Test instance consuming excessive resources.")
        if resource.get("status") == ResourceStatus.STOPPED.value:
            report_lines.append("  ⚠ Instance is STOPPED. Consider rebooting if needed.")
        if (resource.get("type") == ResourceType.EBS.value
                and resource.get("status") == ResourceStatus.AVAILABLE.value):
            report_lines.append("  ⚠ Volume is UNATTACHED. Incurring charges with no active use.")

        return reward, "\n".join(report_lines)

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _find_resource(self, resource_id: str) -> Optional[Dict]:
        """Find a resource dict by ID."""
        for r in self._state.get("resources", []):
            if r["id"] == resource_id:
                return r
        return None

    def _resolve_alerts_for(self, resource_id: str):
        """Remove alerts referencing a given resource."""
        self._state["alerts"] = [
            a for a in self._state.get("alerts", [])
            if a.get("resource_id") != resource_id
        ]

    def _recalculate_state(self):
        """Recalculate derived state fields after an action."""
        # Total hourly cost
        self._state["total_hourly_cost"] = round(
            sum(r.get("cost_per_hour", 0) for r in self._state.get("resources", [])),
            4
        )

        # System uptime based on active alerts
        critical_alerts = [
            a for a in self._state.get("alerts", [])
            if a.get("severity") == "critical"
        ]
        if not critical_alerts:
            self._state["system_uptime"] = min(100.0, self._state.get("system_uptime", 100) + 10.0)
        else:
            self._state["system_uptime"] = max(0.0, self._state.get("system_uptime", 100) - 5.0)

    def _build_alert(self, alert_dict: Dict) -> Any:
        """Build an Alert model from a dict."""
        from models import Alert, AlertSeverity
        return Alert(
            alert_id=alert_dict.get("alert_id", ""),
            severity=AlertSeverity(alert_dict.get("severity", "info")),
            message=alert_dict.get("message", ""),
            resource_id=alert_dict.get("resource_id"),
            metric_name=alert_dict.get("metric_name"),
            metric_value=alert_dict.get("metric_value"),
        )

    # ─── Utility ──────────────────────────────────────────────────────────

    def get_action_history(self) -> List[Dict]:
        """Return the full action history for the current episode."""
        return self._action_history.copy()

    def get_task_description(self) -> str:
        """Return the description of the currently loaded task."""
        if self._task_class:
            return self._task_class.DESCRIPTION
        return "No task loaded."

    def __repr__(self):
        return (
            f"CloudSREEnv(task={self._task_id}, "
            f"step={self._current_step}/{self.max_steps}, "
            f"done={self._done})"
        )
