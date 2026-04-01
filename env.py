"""
Cloud SRE OpenEnv Environment.
Implements the full OpenEnv spec: step(), reset(), state().
Supports seeded procedural generation and chaos event injection.
"""

import copy
import random
from typing import Tuple, Dict, Any, List, Optional
from models import (
    Observation, Action, StepResult, ActionCommand,
    Resource, ResourceStatus, ResourceType
)
from tasks import get_task, Task2LatencySpikeRemediation  # type: ignore


# ─── Chaos Events ─────────────────────────────────────────────────────────────

CHAOS_EVENTS = [
    {
        "type": "new_alert",
        "alert": {
            "alert_id": "chaos-cost-spike",
            "severity": "warning",
            "message": "Unexpected S3 egress cost spike detected: +$0.45/hr from cross-region transfers.",
            "metric_name": "S3EgressCost",
            "metric_value": 0.45,
        },
    },
    {
        "type": "cpu_spike",
        "description": "A random running instance's CPU spikes to 92%",
    },
    {
        "type": "new_alert",
        "alert": {
            "alert_id": "chaos-disk-warn",
            "severity": "warning",
            "message": "Disk utilization on a data volume has reached 85%. Consider expanding storage.",
            "metric_name": "DiskUtilization",
            "metric_value": 85.0,
        },
    },
    {
        "type": "cost_drift",
        "description": "Total cost drifts up slightly due to network egress charges",
    },
]


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
        self._seed: Optional[int] = None
        self._chaos_enabled: bool = False
        self._cost_history: List[float] = []
        self._uptime_history: List[float] = []

    # ─── OpenEnv API ──────────────────────────────────────────────────────

    def reset(self, task_id: str = "phantom_volume_cleanup", seed: Optional[int] = None) -> Observation:
        """
        Reset the environment to a specific task's initial state.

        Args:
            task_id: One of the registered task IDs.
            seed: Optional RNG seed for procedural generation. None = fixed state.

        Returns:
            Initial Observation for the task.
        """
        self._task_class = get_task(task_id)
        self._task_id = task_id
        self._seed = seed

        # Call get_initial_state with seed
        self._initial_state = self._task_class.get_initial_state(seed=seed)
        self._state = copy.deepcopy(self._initial_state)
        self._action_history = []
        self._current_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._chaos_enabled = seed is not None

        # Track cost/uptime history for charts
        self._cost_history = [self._state.get("total_hourly_cost", 0.0)]
        self._uptime_history = [self._state.get("system_uptime", 100.0)]

        return self.state()

    def state(self) -> Observation:
        """Returns the current observation of the environment."""
        return Observation(
            resources=[Resource(**r) for r in self._state.get("resources", [])],
            alerts=[self._build_alert(a) for a in self._state.get("alerts", [])],
            total_hourly_cost=self._state.get("total_hourly_cost", 0.0),
            system_uptime=self._state.get("system_uptime", 100.0),
            step_number=self._current_step,
            max_steps=self.max_steps,
            budget_limit=self._state.get("budget_limit"),
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute a single action in the environment.
        Returns StepResult containing (observation, reward, done, info).
        """
        if self._done:
            return StepResult(
                observation=self.state(), reward=0.0, done=True,
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
            step_reward = -0.01
            info["action_result"] = "Waited one step. No changes made."
        else:
            info["action_result"] = f"Unknown command: {action.command}"

        # ── Record action ──
        self._action_history.append({
            "step": self._current_step,
            "command": action.command.value if isinstance(action.command, ActionCommand) else action.command,
            "resource_id": action.resource_id,
            "params": action.params,
        })

        # ── Inject chaos events ──
        chaos_msg = self._maybe_inject_chaos()
        if chaos_msg:
            info["chaos_event"] = chaos_msg

        # ── Recalculate ──
        self._recalculate_state()

        # ── Track history ──
        self._cost_history.append(self._state.get("total_hourly_cost", 0.0))
        self._uptime_history.append(self._state.get("system_uptime", 100.0))

        # ── Check termination ──
        if self._current_step >= self.max_steps:
            self._done = True
            info["termination_reason"] = "max_steps_reached"

        # ── Grade ──
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
        if self._task_class is None:
            return 0.0, {"error": "No task loaded. Call reset() first."}
        return self._task_class.grade(
            self._action_history, self._state, self._initial_state
        )

    # ─── Virtual Tools (for ReAct agent — do NOT consume a step) ──────────

    def analyze_costs(self) -> str:
        """Analyze cost breakdown by resource type and identify waste."""
        resources = self._state.get("resources", [])
        by_type: Dict[str, float] = {}
        by_env: Dict[str, float] = {}
        waste_candidates = []

        for r in resources:
            rtype = r.get("type", "unknown")
            by_type[rtype] = by_type.get(rtype, 0) + r.get("cost_per_hour", 0)
            env = r.get("tags", {}).get("env", "unknown")
            by_env[env] = by_env.get(env, 0) + r.get("cost_per_hour", 0)

            # Identify waste
            if r.get("status") == "available" and r.get("type") == "ebs_volume":
                waste_candidates.append(f"  - {r['id']}: unattached EBS, ${r.get('cost_per_hour',0):.4f}/hr")
            elif r.get("tags", {}).get("env") == "test" and r.get("cpu_utilization", 0) > 80:
                waste_candidates.append(f"  - {r['id']}: test instance with {r.get('cpu_utilization',0):.0f}% CPU")

        total = sum(by_type.values())
        lines = [
            "=== Cost Analysis Report ===",
            f"Total Hourly Cost: ${total:.4f}/hr",
            "",
            "By Resource Type:",
        ]
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            pct = (c / total * 100) if total > 0 else 0
            lines.append(f"  {t}: ${c:.4f}/hr ({pct:.0f}%)")
        lines.append("")
        lines.append("By Environment:")
        for e, c in sorted(by_env.items(), key=lambda x: -x[1]):
            lines.append(f"  {e}: ${c:.4f}/hr")

        if waste_candidates:
            lines.append("")
            lines.append("Potential Waste:")
            lines.extend(waste_candidates)

        return "\n".join(lines)

    def check_alerts(self) -> str:
        """Summarize all active alerts with context."""
        alerts = self._state.get("alerts", [])
        if not alerts:
            return "No active alerts. All systems nominal."
        lines = [f"=== {len(alerts)} Active Alert(s) ===", ""]
        for a in alerts:
            sev = a.get("severity", "info").upper()
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(sev, "⚪")
            lines.append(f"{icon} [{sev}] {a.get('message', '')}")
            if a.get("resource_id"):
                lines.append(f"   Affected Resource: {a['resource_id']}")
            if a.get("metric_name"):
                lines.append(f"   Metric: {a['metric_name']} = {a.get('metric_value', 'N/A')}")
            lines.append("")
        return "\n".join(lines)

    # ─── Chaos Injection ──────────────────────────────────────────────────

    def _maybe_inject_chaos(self) -> Optional[str]:
        """Inject chaos events based on step and seed. Returns description or None."""
        if not self._chaos_enabled or self._seed is None:
            return None

        rng = random.Random(self._seed * 1000 + self._current_step)
        # Only inject on certain steps and with probability
        if self._current_step < 4 or rng.random() > 0.25:
            return None

        event = rng.choice(CHAOS_EVENTS)

        if event["type"] == "new_alert":
            alert = event["alert"].copy()
            alert["alert_id"] = f"{alert['alert_id']}-step{self._current_step}"
            self._state.setdefault("alerts", []).append(alert)
            return f"⚡ Chaos: New alert injected — {alert['message']}"

        elif event["type"] == "cpu_spike":
            running = [r for r in self._state.get("resources", [])
                       if r.get("status") == "running" and r.get("cpu_utilization", 0) < 80]
            if running:
                target = rng.choice(running)
                target["cpu_utilization"] = round(rng.uniform(88, 96), 1)
                return f"⚡ Chaos: CPU spike on '{target['id']}' → {target['cpu_utilization']}%"

        elif event["type"] == "cost_drift":
            drift = round(rng.uniform(0.05, 0.20), 4)
            self._state["total_hourly_cost"] = round(
                self._state.get("total_hourly_cost", 0) + drift, 4
            )
            return f"⚡ Chaos: Network egress cost drift +${drift:.4f}/hr"

        return None

    # ─── Action Handlers ──────────────────────────────────────────────────

    def _handle_terminate(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.05, "Error: No resource_id provided for terminate."
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        tags = resource.get("tags", {})
        cost = resource.get("cost_per_hour", 0)
        is_prod = tags.get("env") == "prod"
        is_attached_ebs = (
            resource.get("type") == ResourceType.EBS.value
            and resource.get("status") == ResourceStatus.IN_USE.value
        )

        reward = 0.0
        if is_prod and resource.get("type") == ResourceType.EC2.value:
            reward = -0.15
        elif is_attached_ebs:
            reward = -0.10
        elif resource.get("status") == ResourceStatus.AVAILABLE.value:
            reward = 0.05 + (cost * 0.02)
        else:
            reward = 0.02

        self._state["resources"] = [
            r for r in self._state["resources"] if r["id"] != resource_id
        ]
        self._resolve_alerts_for(resource_id)
        return reward, f"Terminated resource '{resource_id}'. Cost saved: ${cost:.4f}/hr."

    def _handle_scale(self, resource_id: Optional[str], target_size: str) -> Tuple[float, str]:
        if not resource_id:
            return -0.05, "Error: No resource_id provided for scale."
        if not target_size:
            return -0.05, "Error: No target_size in params for scale."

        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        old_size = resource.get("instance_size", "unknown")
        old_cost = resource.get("cost_per_hour", 0)

        if resource.get("type") == ResourceType.RDS.value:
            pricing = Task2LatencySpikeRemediation.RDS_PRICING
            if target_size not in pricing:
                return -0.05, f"Error: Invalid RDS size '{target_size}'. Valid: {list(pricing.keys())}"
            new_cost = pricing[target_size]
        else:
            new_cost = old_cost * 2.0

        for r in self._state["resources"]:
            if r["id"] == resource_id:
                r["instance_size"] = target_size
                r["cost_per_hour"] = new_cost
                if r.get("cpu_utilization", 0) > 80:
                    r["cpu_utilization"] = 45.0
                    r["memory_utilization"] = min(r.get("memory_utilization", 50), 60.0)
                break

        self._resolve_alerts_for(resource_id)

        reward = 0.08
        if resource.get("cpu_utilization", 0) > 80:
            reward += 0.05

        return reward, (
            f"Scaled '{resource_id}' from {old_size} to {target_size}. "
            f"Cost changed: ${old_cost:.4f}/hr -> ${new_cost:.4f}/hr."
        )

    def _handle_reboot(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.05, "Error: No resource_id provided for reboot."
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.05, f"Error: Resource '{resource_id}' not found."

        for r in self._state["resources"]:
            if r["id"] == resource_id:
                if r["status"] == ResourceStatus.STOPPED.value:
                    r["status"] = ResourceStatus.RUNNING.value
                    r["cpu_utilization"] = 15.0
                    r["memory_utilization"] = 20.0
                    self._state["system_uptime"] = min(
                        100.0, self._state.get("system_uptime", 0) + 30.0
                    )
                    self._resolve_alerts_for(resource_id)
                    return 0.10, f"Rebooted '{resource_id}'. Instance is now RUNNING."
                elif r["status"] == ResourceStatus.RUNNING.value:
                    r["status"] = ResourceStatus.RUNNING.value
                    r["cpu_utilization"] = 10.0
                    return -0.02, f"Rebooted '{resource_id}'. Temporary disruption."
                else:
                    return -0.05, f"Cannot reboot '{resource_id}' in state '{r['status']}'."

        return -0.05, f"Error: Resource '{resource_id}' not found."

    def _handle_inspect(self, resource_id: Optional[str]) -> Tuple[float, str]:
        if not resource_id:
            return -0.01, "Error: No resource_id provided for inspect."
        resource = self._find_resource(resource_id)
        if resource is None:
            return -0.01, f"Error: Resource '{resource_id}' not found."

        reward = 0.01
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

        cpu = resource.get("cpu_utilization", 0)
        if cpu >= 95:
            report_lines.append("  !! ALERT: CPU utilization critically high!")
        if tags.get("env") == "test" and cpu > 80:
            report_lines.append("  !! WARNING: Test instance consuming excessive resources.")
        if resource.get("status") == ResourceStatus.STOPPED.value:
            report_lines.append("  !! Instance is STOPPED. Consider rebooting if needed.")
        if (resource.get("type") == ResourceType.EBS.value
                and resource.get("status") == ResourceStatus.AVAILABLE.value):
            report_lines.append("  !! Volume is UNATTACHED. Incurring charges with no active use.")

        return reward, "\n".join(report_lines)

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _find_resource(self, resource_id: str) -> Optional[Dict]:
        for r in self._state.get("resources", []):
            if r["id"] == resource_id:
                return r
        return None

    def _resolve_alerts_for(self, resource_id: str):
        self._state["alerts"] = [
            a for a in self._state.get("alerts", [])
            if a.get("resource_id") != resource_id
        ]

    def _recalculate_state(self):
        self._state["total_hourly_cost"] = round(
            sum(r.get("cost_per_hour", 0) for r in self._state.get("resources", [])), 4
        )
        critical_alerts = [
            a for a in self._state.get("alerts", []) if a.get("severity") == "critical"
        ]
        if not critical_alerts:
            self._state["system_uptime"] = min(100.0, self._state.get("system_uptime", 100) + 10.0)
        else:
            self._state["system_uptime"] = max(0.0, self._state.get("system_uptime", 100) - 5.0)

    def _build_alert(self, alert_dict: Dict) -> Any:
        from models import Alert, AlertSeverity
        return Alert(
            alert_id=alert_dict.get("alert_id", ""),
            severity=AlertSeverity(alert_dict.get("severity", "info")),
            message=alert_dict.get("message", ""),
            resource_id=alert_dict.get("resource_id"),
            metric_name=alert_dict.get("metric_name"),
            metric_value=alert_dict.get("metric_value"),
        )

    # ─── Accessors ────────────────────────────────────────────────────────

    def get_action_history(self) -> List[Dict]:
        return self._action_history.copy()

    def get_cost_history(self) -> List[float]:
        return self._cost_history.copy()

    def get_uptime_history(self) -> List[float]:
        return self._uptime_history.copy()

    def get_task_description(self) -> str:
        if self._task_class:
            return self._task_class.DESCRIPTION
        return "No task loaded."

    def __repr__(self):
        return (
            f"CloudSREEnv(task={self._task_id}, "
            f"step={self._current_step}/{self.max_steps}, "
            f"done={self._done})"
        )
