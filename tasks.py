"""
Task definitions and deterministic graders for the Cloud SRE OpenEnv.
Each task provides:
  - get_initial_state(seed=None) -> dict  (the raw state dict loaded into env)
  - grade(action_history, final_state, initial_state) -> (score, breakdown)

Supports seeded procedural generation for randomized scenarios.
"""

import random
from models import (  # type: ignore
    Resource, Alert, ResourceType, ResourceStatus, AlertSeverity
)
from typing import List, Dict, Any, Tuple, Optional, Set


# ── Helper: Name generators ──────────────────────────────────────────────────

_ADJECTIVES = ["old", "legacy", "temp", "stale", "orphan", "unused", "leftover", "backup", "scratch", "test"]
_EC2_ROLES = ["web", "api", "worker", "cache", "monitor", "proxy", "gateway", "scheduler", "indexer", "renderer"]
_EBS_NOTES = ["migration-2024", "snapshot-leftover", "backup-failed", "scratch-disk", "dev-test", "canary-old"]
_ENVS_DECOY = ["deprecated", "test", "dev", "staging", "sandbox"]
_SIZES_EC2 = ["t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge"]
_SIZES_LARGE = ["c5.xlarge", "c5.2xlarge", "c5.4xlarge", "m5.xlarge", "m5.2xlarge"]


def _rand_id(rng: random.Random, prefix: str, width: int = 3) -> str:
    return f"{prefix}-{rng.randint(10**(width-1), 10**width - 1)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — EASY: Phantom Volume Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class Task1PhantomVolumeCleanup:
    """
    Scenario: The cluster has unattached EBS volumes wasting money.
    The agent must identify and terminate them without touching active resources.
    Procedural generation randomises counts, IDs, costs, and decoys.
    """

    TASK_ID = "phantom_volume_cleanup"
    DIFFICULTY = "easy"
    DESCRIPTION = (
        "Your cloud account has unattached EBS volumes that are not connected "
        "to any instance but still incur charges. Identify and terminate them "
        "to reduce costs. Do NOT touch any running instances or in-use volumes."
    )

    # Instance-level storage for the current episode's target IDs
    _current_orphan_ids: Set[str] = set()

    # Legacy fixed IDs kept for backward compatibility with heuristic agent
    UNATTACHED_IDS = {"ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"}

    @classmethod
    def get_initial_state(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            return cls._get_fixed_state()

        rng = random.Random(seed)

        # Randomise counts
        num_orphans = rng.randint(2, 5)
        num_ec2 = rng.randint(3, 6)
        num_inuse_ebs = rng.randint(1, 3)

        resources = []
        orphan_ids: Set[str] = set()

        # Active EC2 instances (decoys — must NOT be terminated)
        ec2_ids = []
        for i in range(num_ec2):
            role = rng.choice(_EC2_ROLES)
            size = rng.choice(_SIZES_EC2)
            eid = _rand_id(rng, f"ec2-{role}")
            ec2_ids.append(eid)
            resources.append(Resource(
                id=eid,
                name=f"{role}-server-{i+1}",
                type=ResourceType.EC2,
                status=ResourceStatus.RUNNING,
                instance_size=size,
                cpu_utilization=round(rng.uniform(10, 75), 1),
                memory_utilization=round(rng.uniform(20, 80), 1),
                cost_per_hour=round(rng.uniform(0.02, 0.12), 4),
                tags={"env": "prod", "role": role},
            ))

        # In-use EBS volumes (decoys)
        for i in range(num_inuse_ebs):
            attached = rng.choice(ec2_ids) if ec2_ids else "ec2-unknown"
            resources.append(Resource(
                id=_rand_id(rng, "ebs-data"),
                name=f"data-vol-{i+1}",
                type=ResourceType.EBS,
                status=ResourceStatus.IN_USE,
                cost_per_hour=round(rng.uniform(0.05, 0.15), 4),
                attached_to=attached,
                tags={"env": "prod"},
            ))

        # ORPHAN EBS volumes (targets)
        for i in range(num_orphans):
            oid = _rand_id(rng, "ebs-orphan")
            orphan_ids.add(oid)
            resources.append(Resource(
                id=oid,
                name=f"{rng.choice(_ADJECTIVES)}-vol-{i+1}",
                type=ResourceType.EBS,
                status=ResourceStatus.AVAILABLE,
                cost_per_hour=round(rng.uniform(0.50, 2.50), 2),
                attached_to=None,
                tags={"env": rng.choice(_ENVS_DECOY), "note": rng.choice(_EBS_NOTES)},
            ))

        # Shuffle to avoid positional bias
        rng.shuffle(resources)

        cls._current_orphan_ids = orphan_ids
        total_waste = sum(r.cost_per_hour for r in resources
                         if r.type == ResourceType.EBS and r.status == ResourceStatus.AVAILABLE)
        total_cost = sum(r.cost_per_hour for r in resources)

        alerts = [Alert(
            alert_id="alert-cost-001",
            severity=AlertSeverity.WARNING,
            message=f"Monthly cost projection exceeds budget. "
                    f"{num_orphans} unattached EBS volumes detected (${total_waste:.2f}/hr waste).",
            metric_name="CostAnomaly",
            metric_value=round(total_waste, 2),
        )]

        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 100.0,
            "budget_limit": None,
        }

    @staticmethod
    def _get_fixed_state() -> Dict[str, Any]:
        """Original fixed state for backward compatibility."""
        resources = [
            Resource(id="ec2-web-001", name="web-server-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=45.0, memory_utilization=62.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "web"}),
            Resource(id="ec2-web-002", name="web-server-2", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=38.0, memory_utilization=55.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "web"}),
            Resource(id="ec2-api-001", name="api-server-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.large",
                     cpu_utilization=60.0, memory_utilization=70.0,
                     cost_per_hour=0.0832, tags={"env": "prod", "role": "api"}),
            Resource(id="ec2-worker-001", name="bg-worker-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.small",
                     cpu_utilization=20.0, memory_utilization=30.0,
                     cost_per_hour=0.0208, tags={"env": "prod", "role": "worker"}),
            Resource(id="ec2-worker-002", name="bg-worker-2", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.small",
                     cpu_utilization=15.0, memory_utilization=25.0,
                     cost_per_hour=0.0208, tags={"env": "prod", "role": "worker"}),
            Resource(id="ebs-data-001", name="web-data-vol", type=ResourceType.EBS,
                     status=ResourceStatus.IN_USE, cost_per_hour=0.10,
                     attached_to="ec2-web-001", tags={"env": "prod"}),
            Resource(id="ebs-data-002", name="api-data-vol", type=ResourceType.EBS,
                     status=ResourceStatus.IN_USE, cost_per_hour=0.10,
                     attached_to="ec2-api-001", tags={"env": "prod"}),
            Resource(id="ebs-orphan-001", name="old-migration-vol", type=ResourceType.EBS,
                     status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                     attached_to=None, tags={"env": "deprecated", "note": "migration-2024"}),
            Resource(id="ebs-orphan-002", name="test-snapshot-vol", type=ResourceType.EBS,
                     status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                     attached_to=None, tags={"env": "test", "note": "snapshot-leftover"}),
            Resource(id="ebs-orphan-003", name="dev-scratch-vol", type=ResourceType.EBS,
                     status=ResourceStatus.AVAILABLE, cost_per_hour=1.40,
                     attached_to=None, tags={"env": "dev", "note": "scratch-disk"}),
        ]
        alerts = [Alert(
            alert_id="alert-cost-001", severity=AlertSeverity.WARNING,
            message="Monthly cost projection exceeds budget by 15%. "
                    "3 unattached EBS volumes detected ($4.20/hr waste).",
            metric_name="CostAnomaly", metric_value=4.20,
        )]
        total_cost = sum(r.cost_per_hour for r in resources)
        Task1PhantomVolumeCleanup._current_orphan_ids = {"ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"}
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 100.0,
            "budget_limit": None,
        }

    @classmethod
    def grade(cls, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        target_ids = cls._current_orphan_ids if cls._current_orphan_ids else cls.UNATTACHED_IDS
        score = 0.0
        breakdown = {
            "orphans_terminated": [],
            "active_resources_terminated": [],
            "total_orphans": len(target_ids),
            "penalties": 0.0,
        }
        terminated_ids = set()
        for entry in action_history:
            if entry.get("command") == "terminate" and entry.get("resource_id"):
                terminated_ids.add(entry["resource_id"])
        per_orphan = 1.0 / len(target_ids) if target_ids else 0.33
        for rid in terminated_ids:
            if rid in target_ids:
                score += per_orphan
                breakdown["orphans_terminated"].append(rid)
            else:
                score -= 0.50
                breakdown["active_resources_terminated"].append(rid)
                breakdown["penalties"] += 0.50
        score = round(max(0.01, min(0.99, score)), 2)
        if len(breakdown["orphans_terminated"]) == len(target_ids) and not breakdown["active_resources_terminated"]:
            score = 0.99
        return score, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — MEDIUM: Latency Spike Remediation
# ═══════════════════════════════════════════════════════════════════════════════

class Task2LatencySpikeRemediation:
    """
    Scenario: An RDS database is under-provisioned causing high API latency.
    The agent must scale it up while staying under budget.
    """

    TASK_ID = "latency_spike_remediation"
    DIFFICULTY = "medium"
    DESCRIPTION = (
        "A critical latency alert has fired. Your RDS database is running on a "
        "tiny instance and cannot handle the current load. Scale it up to fix "
        "latency, but keep total hourly cost under the budget limit. "
        "Do NOT terminate running instances to free up budget."
    )

    TARGET_RDS_ID = "rds-primary-001"
    BUDGET_LIMIT = 12.00
    _current_rds_id: str = "rds-primary-001"

    RDS_PRICING = {
        "db.t3.micro": 0.017, "db.t3.small": 0.034,
        "db.t3.medium": 0.068, "db.t3.large": 0.136,
        "db.t3.xlarge": 0.272,
    }

    @classmethod
    def get_initial_state(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            cls._current_rds_id = cls.TARGET_RDS_ID
            return cls._get_fixed_state()

        rng = random.Random(seed)
        num_ec2 = rng.randint(3, 6)
        resources = []

        for i in range(num_ec2):
            role = rng.choice(["app", "cache", "monitor", "proxy", "gateway"])
            resources.append(Resource(
                id=_rand_id(rng, f"ec2-{role}"),
                name=f"{role}-server-{i+1}",
                type=ResourceType.EC2,
                status=ResourceStatus.RUNNING,
                instance_size=rng.choice(_SIZES_EC2[2:]),  # medium+
                cpu_utilization=round(rng.uniform(25, 75), 1),
                memory_utilization=round(rng.uniform(30, 80), 1),
                cost_per_hour=round(rng.uniform(0.04, 0.15), 4),
                tags={"env": "prod", "role": role},
            ))

        # Under-provisioned RDS (the problem)
        rds_id = _rand_id(rng, "rds-primary")
        cls._current_rds_id = rds_id
        rds_cpu = round(rng.uniform(92, 99), 1)
        resources.append(Resource(
            id=rds_id, name="primary-db", type=ResourceType.RDS,
            status=ResourceStatus.RUNNING, instance_size="db.t3.micro",
            cpu_utilization=rds_cpu, memory_utilization=round(rng.uniform(88, 98), 1),
            cost_per_hour=0.017, tags={"env": "prod", "role": "database"},
        ))

        # ALB
        resources.append(Resource(
            id=_rand_id(rng, "alb-main"), name="main-load-balancer",
            type=ResourceType.ALB, status=ResourceStatus.RUNNING,
            cost_per_hour=0.0225, tags={"env": "prod", "role": "lb"},
        ))

        rng.shuffle(resources)

        latency_val = round(rng.uniform(1800, 3000), 0)
        alerts = [
            Alert(alert_id="alert-latency-001", severity=AlertSeverity.CRITICAL,
                  message=f"API p99 latency has exceeded {latency_val:.0f}ms. "
                          f"Root cause: RDS '{rds_id}' (db.t3.micro) at {rds_cpu}% CPU.",
                  resource_id=rds_id, metric_name="P99Latency", metric_value=latency_val),
            Alert(alert_id="alert-cpu-001", severity=AlertSeverity.WARNING,
                  message=f"RDS '{rds_id}' CPU utilization is at {rds_cpu}%.",
                  resource_id=rds_id, metric_name="CPUUtilization", metric_value=rds_cpu),
        ]

        total_cost = sum(r.cost_per_hour for r in resources)
        budget = round(total_cost + rng.uniform(3.0, 8.0), 2)

        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": round(rng.uniform(65, 82), 1),
            "budget_limit": budget,
        }

    @staticmethod
    def _get_fixed_state() -> Dict[str, Any]:
        """Original fixed state."""
        resources = [
            Resource(id="ec2-app-001", name="app-server-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.large",
                     cpu_utilization=72.0, memory_utilization=65.0,
                     cost_per_hour=0.0832, tags={"env": "prod", "role": "app"}),
            Resource(id="ec2-app-002", name="app-server-2", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.large",
                     cpu_utilization=68.0, memory_utilization=60.0,
                     cost_per_hour=0.0832, tags={"env": "prod", "role": "app"}),
            Resource(id="ec2-cache-001", name="redis-cache-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="r5.large",
                     cpu_utilization=30.0, memory_utilization=80.0,
                     cost_per_hour=0.126, tags={"env": "prod", "role": "cache"}),
            Resource(id="ec2-monitor-001", name="monitoring-node", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=10.0, memory_utilization=20.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "monitoring"}),
            Resource(id="rds-primary-001", name="primary-db", type=ResourceType.RDS,
                     status=ResourceStatus.RUNNING, instance_size="db.t3.micro",
                     cpu_utilization=98.0, memory_utilization=95.0,
                     cost_per_hour=0.017, tags={"env": "prod", "role": "database"}),
            Resource(id="alb-main-001", name="main-load-balancer", type=ResourceType.ALB,
                     status=ResourceStatus.RUNNING,
                     cost_per_hour=0.0225, tags={"env": "prod", "role": "lb"}),
        ]
        alerts = [
            Alert(alert_id="alert-latency-001", severity=AlertSeverity.CRITICAL,
                  message="API p99 latency has exceeded 2000ms for the last 10 minutes. "
                          "Root cause: RDS 'primary-db' (db.t3.micro) at 98% CPU.",
                  resource_id="rds-primary-001", metric_name="P99Latency", metric_value=2150.0),
            Alert(alert_id="alert-cpu-001", severity=AlertSeverity.WARNING,
                  message="RDS 'primary-db' CPU utilization is at 98%.",
                  resource_id="rds-primary-001", metric_name="CPUUtilization", metric_value=98.0),
        ]
        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 78.0,
            "budget_limit": Task2LatencySpikeRemediation.BUDGET_LIMIT,
        }

    @classmethod
    def grade(cls, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        target_rds = cls._current_rds_id
        score = 0.0
        breakdown = {
            "rds_scaled": False, "under_budget": False,
            "alert_resolved": False, "ec2s_terminated": [], "penalties": 0.0,
        }
        valid_sizes = ["db.t3.medium", "db.t3.large", "db.t3.xlarge"]
        for entry in action_history:
            cmd, rid = entry.get("command"), entry.get("resource_id")
            params = entry.get("params", {})
            if cmd == "scale" and rid == target_rds:
                if params.get("target_size", "") in valid_sizes:
                    breakdown["rds_scaled"] = True
            if cmd == "terminate" and rid and rid.startswith("ec2"):
                breakdown["ec2s_terminated"].append(rid)

        if breakdown["rds_scaled"]:
            score += 0.40
            breakdown["alert_resolved"] = True
            score += 0.30
        budget = initial_state.get("budget_limit", cls.BUDGET_LIMIT)
        if final_state.get("total_hourly_cost", 999) <= budget:
            breakdown["under_budget"] = True
            score += 0.30
        for _ in breakdown["ec2s_terminated"]:
            score -= 0.30
            breakdown["penalties"] += 0.30
        score = round(max(0.01, min(0.99, score)), 2)
        return score, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — HARD: Noisy Neighbor Incident
# ═══════════════════════════════════════════════════════════════════════════════

class Task3NoisyNeighborIncident:
    """
    Scenario: A rogue test EC2 instance is consuming excessive CPU and has
    crashed a production backend. Agent must investigate, terminate rogue,
    and restore production.
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
    _current_rogue_id: str = "ec2-rogue-test-001"
    _current_backend_id: str = "ec2-backend-prod-001"
    _current_prod_ids: Set[str] = set()

    @classmethod
    def get_initial_state(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is None:
            cls._current_rogue_id = cls.ROGUE_ID
            cls._current_backend_id = cls.BACKEND_ID
            cls._current_prod_ids = {"ec2-frontend-001", "ec2-frontend-002",
                                      "ec2-api-prod-001", "ec2-db-proxy-001",
                                      "ec2-backend-prod-001", "rds-prod-001"}
            return cls._get_fixed_state()

        rng = random.Random(seed)
        resources = []
        prod_ids: Set[str] = set()

        # THE ROGUE INSTANCE
        rogue_id = _rand_id(rng, "ec2-rogue-test")
        cls._current_rogue_id = rogue_id
        rogue_size = rng.choice(_SIZES_LARGE)
        resources.append(Resource(
            id=rogue_id, name="load-test-runner", type=ResourceType.EC2,
            status=ResourceStatus.RUNNING, instance_size=rogue_size,
            cpu_utilization=100.0, memory_utilization=round(rng.uniform(85, 98), 1),
            cost_per_hour=round(rng.uniform(0.40, 1.20), 2),
            tags={"env": "test", "role": "load-testing", "owner": "qa-team", "note": "forgot to stop"},
        ))

        # CRASHED PROD BACKEND
        backend_id = _rand_id(rng, "ec2-backend-prod")
        cls._current_backend_id = backend_id
        prod_ids.add(backend_id)
        resources.append(Resource(
            id=backend_id, name="backend-primary", type=ResourceType.EC2,
            status=ResourceStatus.STOPPED, instance_size="m5.xlarge",
            cpu_utilization=0.0, memory_utilization=0.0,
            cost_per_hour=0.192,
            tags={"env": "prod", "role": "backend", "critical": "true"},
        ))

        # Normal prod instances (decoys)
        num_prod = rng.randint(3, 6)
        for i in range(num_prod):
            role = rng.choice(["frontend", "api", "db-proxy", "gateway", "worker"])
            pid = _rand_id(rng, f"ec2-{role}-prod")
            prod_ids.add(pid)
            resources.append(Resource(
                id=pid, name=f"{role}-{i+1}", type=ResourceType.EC2,
                status=ResourceStatus.RUNNING,
                instance_size=rng.choice(_SIZES_EC2[2:]),
                cpu_utilization=round(rng.uniform(20, 75), 1),
                memory_utilization=round(rng.uniform(25, 70), 1),
                cost_per_hour=round(rng.uniform(0.03, 0.10), 4),
                tags={"env": "prod", "role": role},
            ))

        # RDS
        rds_id = _rand_id(rng, "rds-prod")
        prod_ids.add(rds_id)
        resources.append(Resource(
            id=rds_id, name="prod-database", type=ResourceType.RDS,
            status=ResourceStatus.RUNNING, instance_size="db.r5.large",
            cpu_utilization=round(rng.uniform(30, 55), 1),
            memory_utilization=round(rng.uniform(40, 65), 1),
            cost_per_hour=0.24, tags={"env": "prod", "role": "database"},
        ))

        cls._current_prod_ids = prod_ids
        rng.shuffle(resources)

        rogue_cost = next(r.cost_per_hour for r in resources if r.id == rogue_id)
        alerts = [
            Alert(alert_id="alert-crit-001", severity=AlertSeverity.CRITICAL,
                  message=f"Production backend '{backend_id}' is DOWN. HTTP 503 errors spiking.",
                  resource_id=backend_id, metric_name="HealthCheck", metric_value=0.0),
            Alert(alert_id="alert-crit-002", severity=AlertSeverity.CRITICAL,
                  message=f"Abnormal CPU usage: '{rogue_id}' consuming 100% CPU on {rogue_size}. Investigate immediately.",
                  resource_id=rogue_id, metric_name="CPUUtilization", metric_value=100.0),
            Alert(alert_id="alert-cost-002", severity=AlertSeverity.WARNING,
                  message=f"Hourly cost spike: ${rogue_cost:.2f}/hr from a single test instance.",
                  resource_id=rogue_id, metric_name="CostAnomaly", metric_value=rogue_cost),
        ]

        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": round(rng.uniform(25, 45), 1),
            "budget_limit": None,
        }

    @staticmethod
    def _get_fixed_state() -> Dict[str, Any]:
        """Original fixed state."""
        resources = [
            Resource(id="ec2-rogue-test-001", name="load-test-runner", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="c5.4xlarge",
                     cpu_utilization=100.0, memory_utilization=92.0,
                     cost_per_hour=0.68, tags={"env": "test", "role": "load-testing",
                                                "owner": "qa-team", "note": "forgot to stop"}),
            Resource(id="ec2-backend-prod-001", name="backend-primary", type=ResourceType.EC2,
                     status=ResourceStatus.STOPPED, instance_size="m5.xlarge",
                     cpu_utilization=0.0, memory_utilization=0.0,
                     cost_per_hour=0.192,
                     tags={"env": "prod", "role": "backend", "critical": "true"}),
            Resource(id="ec2-frontend-001", name="frontend-1", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=55.0, memory_utilization=40.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "frontend"}),
            Resource(id="ec2-frontend-002", name="frontend-2", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=50.0, memory_utilization=38.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "frontend"}),
            Resource(id="ec2-api-prod-001", name="api-gateway", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.large",
                     cpu_utilization=75.0, memory_utilization=60.0,
                     cost_per_hour=0.0832, tags={"env": "prod", "role": "api"}),
            Resource(id="ec2-db-proxy-001", name="db-proxy", type=ResourceType.EC2,
                     status=ResourceStatus.RUNNING, instance_size="t3.medium",
                     cpu_utilization=30.0, memory_utilization=45.0,
                     cost_per_hour=0.0416, tags={"env": "prod", "role": "db-proxy"}),
            Resource(id="rds-prod-001", name="prod-database", type=ResourceType.RDS,
                     status=ResourceStatus.RUNNING, instance_size="db.r5.large",
                     cpu_utilization=40.0, memory_utilization=55.0,
                     cost_per_hour=0.24, tags={"env": "prod", "role": "database"}),
        ]
        alerts = [
            Alert(alert_id="alert-crit-001", severity=AlertSeverity.CRITICAL,
                  message="Production backend 'backend-primary' is DOWN. HTTP 503 errors spiking.",
                  resource_id="ec2-backend-prod-001", metric_name="HealthCheck", metric_value=0.0),
            Alert(alert_id="alert-crit-002", severity=AlertSeverity.CRITICAL,
                  message="Abnormal CPU usage: 'ec2-rogue-test-001' consuming 100% CPU on c5.4xlarge. Investigate immediately.",
                  resource_id="ec2-rogue-test-001", metric_name="CPUUtilization", metric_value=100.0),
            Alert(alert_id="alert-cost-002", severity=AlertSeverity.WARNING,
                  message="Hourly cost spike: $0.68/hr from a single test instance.",
                  resource_id="ec2-rogue-test-001", metric_name="CostAnomaly", metric_value=0.68),
        ]
        total_cost = sum(r.cost_per_hour for r in resources)
        return {
            "resources": [r.model_dump() for r in resources],
            "alerts": [a.model_dump() for a in alerts],
            "total_hourly_cost": round(total_cost, 4),
            "system_uptime": 35.0,
            "budget_limit": None,
        }

    @classmethod
    def grade(cls, action_history: List[Dict], final_state: Dict, initial_state: Dict) -> Tuple[float, Dict]:
        rogue_id = cls._current_rogue_id
        backend_id = cls._current_backend_id
        prod_ids = cls._current_prod_ids

        score = 0.0
        breakdown = {
            "inspected_rogue": False, "terminated_rogue": False,
            "rebooted_backend": False, "alerts_resolved": False,
            "prod_terminated": [], "penalties": 0.0,
        }

        has_inspected_rogue = False
        inspected_before_terminate = False

        for entry in action_history:
            cmd, rid = entry.get("command"), entry.get("resource_id")
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

        score = round(max(0.01, min(0.99, score)), 2)
        return score, breakdown


# ─── Task Registry ────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "phantom_volume_cleanup": Task1PhantomVolumeCleanup,
    "latency_spike_remediation": Task2LatencySpikeRemediation,
    "noisy_neighbor_incident": Task3NoisyNeighborIncident,
}

def get_task(task_id: str):
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]

def list_tasks() -> List[Dict[str, str]]:
    return [
        {"id": cls.TASK_ID, "name": cls.__name__,
         "difficulty": cls.DIFFICULTY, "description": cls.DESCRIPTION}
        for cls in TASK_REGISTRY.values()
    ]
