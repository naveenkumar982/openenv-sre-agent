"""
Task definitions and deterministic graders for the Cloud SRE OpenEnv.
Each task provides:
  - get_initial_state() -> dict  (the raw state dict loaded into env)
  - grade(action_history, final_state, initial_state) -> (score, breakdown)
"""

from models import (  # type: ignore
    Resource, Alert, ResourceType, ResourceStatus, AlertSeverity
)
from typing import List, Dict, Any, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — EASY: Phantom Volume Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class Task1PhantomVolumeCleanup:
    """
    Scenario: The cluster has 3 unattached EBS volumes wasting $4.20/hr.
    The agent must identify and terminate all 3 without touching active resources.
    """

    TASK_ID = "phantom_volume_cleanup"
    DIFFICULTY = "easy"
    DESCRIPTION = (
        "Your cloud account has 3 unattached EBS volumes that are not connected "
        "to any instance but still incur charges. Identify and terminate them "
        "to reduce costs. Do NOT touch any running instances or in-use volumes."
    )

    # IDs of target unattached volumes
    UNATTACHED_IDS = {"ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"}

    @staticmethod
    def get_initial_state() -> Dict[str, Any]:
        resources = [
            # ── Active EC2 Instances ──
            Resource(
                id="ec2-web-001", name="web-server-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=45.0, memory_utilization=62.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "web"}
            ),
            Resource(
                id="ec2-web-002", name="web-server-2", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=38.0, memory_utilization=55.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "web"}
            ),
            Resource(
                id="ec2-api-001", name="api-server-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.large",
                cpu_utilization=60.0, memory_utilization=70.0,
                cost_per_hour=0.0832, tags={"env": "prod", "role": "api"}
            ),
            Resource(
                id="ec2-worker-001", name="bg-worker-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.small",
                cpu_utilization=20.0, memory_utilization=30.0,
                cost_per_hour=0.0208, tags={"env": "prod", "role": "worker"}
            ),
            Resource(
                id="ec2-worker-002", name="bg-worker-2", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.small",
                cpu_utilization=15.0, memory_utilization=25.0,
                cost_per_hour=0.0208, tags={"env": "prod", "role": "worker"}
            ),
            # ── In-use EBS Volumes ──
            Resource(
                id="ebs-data-001", name="web-data-vol", type=ResourceType.EBS,
                status=ResourceStatus.IN_USE, cost_per_hour=0.10,
                attached_to="ec2-web-001", tags={"env": "prod"}
            ),
            Resource(
                id="ebs-data-002", name="api-data-vol", type=ResourceType.EBS,
                status=ResourceStatus.IN_USE, cost_per_hour=0.10,
                attached_to="ec2-api-001", tags={"env": "prod"}
            ),
            # ── ORPHAN EBS Volumes (targets) ──
            Resource(
                id="ebs-orphan-001", name="old-migration-vol", type=ResourceType.EBS,
                status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                attached_to=None, tags={"env": "deprecated", "note": "migration-2024"}
            ),
            Resource(
                id="ebs-orphan-002", name="test-snapshot-vol", type=ResourceType.EBS,
                status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                attached_to=None, tags={"env": "test", "note": "snapshot-leftover"}
            ),
            Resource(
                id="ebs-orphan-003", name="dev-scratch-vol", type=ResourceType.EBS,
                status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                attached_to=None, tags={"env": "dev", "note": "scratch-disk"}
            ),
        ]

        alerts = [
            Alert(
                alert_id="alert-cost-001",
                severity=AlertSeverity.WARNING,
                message="Monthly cost projection exceeds budget by 15%. "
                        "3 unattached EBS volumes detected ($4.20/hr waste).",
                metric_name="CostAnomaly",
                metric_value=4.20
            ),
        ]

        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 100.0,
            "budget_limit": None,
        }

    @staticmethod
    def grade(action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        """
        Grading:
          +0.33 per unattached EBS correctly terminated  (max +1.0)
          -0.50 per active resource accidentally terminated
        """
        score = 0.0
        breakdown = {
            "orphans_terminated": [],
            "active_resources_terminated": [],
            "penalties": 0.0,
        }

        terminated_ids = set()
        for entry in action_history:
            if entry.get("command") == "terminate" and entry.get("resource_id"):
                terminated_ids.add(entry["resource_id"])

        for rid in terminated_ids:
            if rid in Task1PhantomVolumeCleanup.UNATTACHED_IDS:
                score += 0.33
                breakdown["orphans_terminated"].append(rid)
            else:
                score -= 0.50
                breakdown["active_resources_terminated"].append(rid)
                breakdown["penalties"] += 0.50

        # Cap between 0.0 and 1.0, round for precision
        score = round(max(0.0, min(1.0, score)), 2)
        # Adjust: 3 * 0.33 = 0.99, bump to 1.0 if all 3 found
        if len(breakdown["orphans_terminated"]) == 3 and not breakdown["active_resources_terminated"]:
            score = 1.0

        return score, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — MEDIUM: Latency Spike Remediation
# ═══════════════════════════════════════════════════════════════════════════════

class Task2LatencySpikeRemediation:
    """
    Scenario: An RDS database is under-provisioned (db.t3.micro) causing high
    API latency. The agent must scale it up to at least db.t3.medium while
    staying under the $12.00/hr budget. Killing EC2 instances to save budget
    is penalized as destructive behavior.
    """

    TASK_ID = "latency_spike_remediation"
    DIFFICULTY = "medium"
    DESCRIPTION = (
        "A critical latency alert has fired. Your RDS database 'primary-db' is "
        "running on db.t3.micro and cannot handle the current load. Scale it up "
        "to fix latency, but keep total hourly cost under $12.00/hr. "
        "Do NOT terminate running instances to free up budget."
    )

    TARGET_RDS_ID = "rds-primary-001"
    BUDGET_LIMIT = 12.00

    # Pricing table for RDS sizes
    RDS_PRICING = {
        "db.t3.micro": 0.017,
        "db.t3.small": 0.034,
        "db.t3.medium": 0.068,
        "db.t3.large": 0.136,
        "db.t3.xlarge": 0.272,
    }

    @staticmethod
    def get_initial_state() -> Dict[str, Any]:
        resources = [
            # ── EC2 Instances ──
            Resource(
                id="ec2-app-001", name="app-server-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.large",
                cpu_utilization=72.0, memory_utilization=65.0,
                cost_per_hour=0.0832, tags={"env": "prod", "role": "app"}
            ),
            Resource(
                id="ec2-app-002", name="app-server-2", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.large",
                cpu_utilization=68.0, memory_utilization=60.0,
                cost_per_hour=0.0832, tags={"env": "prod", "role": "app"}
            ),
            Resource(
                id="ec2-cache-001", name="redis-cache-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="r5.large",
                cpu_utilization=30.0, memory_utilization=80.0,
                cost_per_hour=0.126, tags={"env": "prod", "role": "cache"}
            ),
            Resource(
                id="ec2-monitor-001", name="monitoring-node", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=10.0, memory_utilization=20.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "monitoring"}
            ),
            # ── Under-provisioned RDS (the problem) ──
            Resource(
                id="rds-primary-001", name="primary-db", type=ResourceType.RDS,
                status=ResourceStatus.RUNNING, instance_size="db.t3.micro",
                cpu_utilization=98.0, memory_utilization=95.0,
                cost_per_hour=0.017, tags={"env": "prod", "role": "database"}
            ),
            # ── ALB ──
            Resource(
                id="alb-main-001", name="main-load-balancer", type=ResourceType.ALB,
                status=ResourceStatus.RUNNING,
                cost_per_hour=0.0225, tags={"env": "prod", "role": "lb"}
            ),
        ]

        alerts = [
            Alert(
                alert_id="alert-latency-001",
                severity=AlertSeverity.CRITICAL,
                message="API p99 latency has exceeded 2000ms for the last 10 minutes. "
                        "Root cause: RDS 'primary-db' (db.t3.micro) at 98% CPU.",
                resource_id="rds-primary-001",
                metric_name="P99Latency",
                metric_value=2150.0
            ),
            Alert(
                alert_id="alert-cpu-001",
                severity=AlertSeverity.WARNING,
                message="RDS 'primary-db' CPU utilization is at 98%.",
                resource_id="rds-primary-001",
                metric_name="CPUUtilization",
                metric_value=98.0
            ),
        ]

        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 78.0,  # Degraded due to latency
            "budget_limit": Task2LatencySpikeRemediation.BUDGET_LIMIT,
        }

    @staticmethod
    def grade(action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        """
        Grading:
          +0.40  RDS scaled to db.t3.medium or larger
          +0.30  Total cost still under budget limit
          +0.30  Latency alert resolved (auto-clears on scale)
          -0.30  Per EC2 instance terminated (destructive shortcut)
        """
        score = 0.0
        breakdown = {
            "rds_scaled": False,
            "under_budget": False,
            "alert_resolved": False,
            "ec2s_terminated": [],
            "penalties": 0.0,
        }

        valid_sizes = ["db.t3.medium", "db.t3.large", "db.t3.xlarge"]

        for entry in action_history:
            cmd = entry.get("command")
            rid = entry.get("resource_id")
            params = entry.get("params", {})

            if cmd == "scale" and rid == Task2LatencySpikeRemediation.TARGET_RDS_ID:
                target = params.get("target_size", "")
                if target in valid_sizes:
                    breakdown["rds_scaled"] = True

            if cmd == "terminate" and rid and rid.startswith("ec2"):
                breakdown["ec2s_terminated"].append(rid)

        # Score components
        if breakdown["rds_scaled"]:
            score += 0.40
            # Alert resolves automatically when RDS is scaled
            breakdown["alert_resolved"] = True
            score += 0.30

        # Check budget in final state
        final_cost = final_state.get("total_hourly_cost", 999)
        if final_cost <= Task2LatencySpikeRemediation.BUDGET_LIMIT:
            breakdown["under_budget"] = True
            score += 0.30

        # Penalties for destructive actions
        for _ in breakdown["ec2s_terminated"]:
            score -= 0.30
            breakdown["penalties"] += 0.30

        score = round(max(0.0, min(1.0, score)), 2)
        return score, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — HARD: Noisy Neighbor Incident
# ═══════════════════════════════════════════════════════════════════════════════

class Task3NoisyNeighborIncident:
    """
    Scenario: A rogue EC2 test instance is consuming excessive CPU, driving up
    costs, and has caused the production backend to crash from resource starvation.
    The agent must:
      1. Inspect the rogue instance to confirm it's the offender
      2. Terminate the rogue instance
      3. Reboot the crashed production backend
    """

    TASK_ID = "noisy_neighbor_incident"
    DIFFICULTY = "hard"
    DESCRIPTION = (
        "CRITICAL INCIDENT: A rogue EC2 instance (tagged env:test) is consuming "
        "excessive resources and has caused the production backend server to crash. "
        "Investigate the situation, terminate the offending instance, and restore "
        "the production backend. Be careful not to terminate any production instances."
    )

    ROGUE_ID = "ec2-rogue-test-001"
    BACKEND_ID = "ec2-backend-prod-001"

    @staticmethod
    def get_initial_state() -> Dict[str, Any]:
        resources = [
            # ── THE ROGUE INSTANCE ──
            Resource(
                id="ec2-rogue-test-001", name="load-test-runner", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="c5.4xlarge",
                cpu_utilization=100.0, memory_utilization=92.0,
                cost_per_hour=0.68, tags={"env": "test", "role": "load-testing",
                                           "owner": "qa-team", "note": "forgot to stop"}
            ),
            # ── CRASHED PROD BACKEND ──
            Resource(
                id="ec2-backend-prod-001", name="backend-primary", type=ResourceType.EC2,
                status=ResourceStatus.STOPPED, instance_size="m5.xlarge",
                cpu_utilization=0.0, memory_utilization=0.0,
                cost_per_hour=0.192,
                tags={"env": "prod", "role": "backend", "critical": "true"}
            ),
            # ── Normal prod instances ──
            Resource(
                id="ec2-frontend-001", name="frontend-1", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=55.0, memory_utilization=40.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "frontend"}
            ),
            Resource(
                id="ec2-frontend-002", name="frontend-2", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=50.0, memory_utilization=38.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "frontend"}
            ),
            Resource(
                id="ec2-api-prod-001", name="api-gateway", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.large",
                cpu_utilization=75.0, memory_utilization=60.0,
                cost_per_hour=0.0832, tags={"env": "prod", "role": "api"}
            ),
            Resource(
                id="ec2-db-proxy-001", name="db-proxy", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING, instance_size="t3.medium",
                cpu_utilization=30.0, memory_utilization=45.0,
                cost_per_hour=0.0416, tags={"env": "prod", "role": "db-proxy"}
            ),
            # ── RDS ──
            Resource(
                id="rds-prod-001", name="prod-database", type=ResourceType.RDS,
                status=ResourceStatus.RUNNING, instance_size="db.r5.large",
                cpu_utilization=40.0, memory_utilization=55.0,
                cost_per_hour=0.24, tags={"env": "prod", "role": "database"}
            ),
        ]

        alerts = [
            Alert(
                alert_id="alert-crit-001",
                severity=AlertSeverity.CRITICAL,
                message="Production backend 'backend-primary' is DOWN. "
                        "HTTP 503 errors spiking across all API endpoints.",
                resource_id="ec2-backend-prod-001",
                metric_name="HealthCheck",
                metric_value=0.0
            ),
            Alert(
                alert_id="alert-crit-002",
                severity=AlertSeverity.CRITICAL,
                message="Abnormal CPU usage detected: instance 'ec2-rogue-test-001' "
                        "consuming 100% CPU on a c5.4xlarge. Investigate immediately.",
                resource_id="ec2-rogue-test-001",
                metric_name="CPUUtilization",
                metric_value=100.0
            ),
            Alert(
                alert_id="alert-cost-002",
                severity=AlertSeverity.WARNING,
                message="Hourly cost spike detected: $0.68/hr from a single test instance.",
                resource_id="ec2-rogue-test-001",
                metric_name="CostAnomaly",
                metric_value=0.68
            ),
        ]

        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 35.0,  # Severely degraded
            "budget_limit": None,
        }

    @staticmethod
    def grade(action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        """
        Grading:
          +0.20  Inspected the rogue instance before acting (reasoning signal)
          +0.30  Terminated the rogue test instance
          +0.30  Rebooted the crashed production backend
          +0.20  All critical alerts resolved (derived from above actions)
          -0.50  Per production instance accidentally terminated
        """
        score = 0.0
        breakdown = {
            "inspected_rogue": False,
            "terminated_rogue": False,
            "rebooted_backend": False,
            "alerts_resolved": False,
            "prod_terminated": [],
            "penalties": 0.0,
        }

        rogue_id = Task3NoisyNeighborIncident.ROGUE_ID
        backend_id = Task3NoisyNeighborIncident.BACKEND_ID
        prod_ids = {"ec2-frontend-001", "ec2-frontend-002",
                     "ec2-api-prod-001", "ec2-db-proxy-001",
                     "ec2-backend-prod-001", "rds-prod-001"}

        inspected_before_terminate = False
        has_inspected_rogue = False

        for entry in action_history:
            cmd = entry.get("command")
            rid = entry.get("resource_id")

            if cmd == "inspect" and rid == rogue_id:
                has_inspected_rogue = True

            if cmd == "terminate" and rid == rogue_id:
                breakdown["terminated_rogue"] = True
                if has_inspected_rogue:
                    inspected_before_terminate = True

            if cmd == "reboot" and rid == backend_id:
                breakdown["rebooted_backend"] = True

            if cmd == "terminate" and rid in prod_ids:
                breakdown["prod_terminated"].append(rid)

        # Scoring
        if inspected_before_terminate:
            breakdown["inspected_rogue"] = True
            score += 0.20

        if breakdown["terminated_rogue"]:
            score += 0.30

        if breakdown["rebooted_backend"]:
            score += 0.30

        if breakdown["terminated_rogue"] and breakdown["rebooted_backend"]:
            breakdown["alerts_resolved"] = True
            score += 0.20

        for _ in breakdown["prod_terminated"]:
            score -= 0.50
            breakdown["penalties"] += 0.50

        score = round(max(0.0, min(1.0, score)), 2)
        return score, breakdown


# ─── Task Registry ────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "phantom_volume_cleanup": Task1PhantomVolumeCleanup,
    "latency_spike_remediation": Task2LatencySpikeRemediation,
    "noisy_neighbor_incident": Task3NoisyNeighborIncident,
}

def get_task(task_id: str):
    """Returns the task class for a given task_id."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]

def list_tasks() -> List[Dict[str, str]]:
    """Returns a summary of all available tasks."""
    return [
        {
            "id": cls.TASK_ID,
            "name": cls.__name__,
            "difficulty": cls.DIFFICULTY,
            "description": cls.DESCRIPTION,
        }
        for cls in TASK_REGISTRY.values()
    ]
