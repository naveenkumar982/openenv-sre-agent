"""
Typed Pydantic models for the Cloud SRE OpenEnv environment.
Defines the Observation, Action, and supporting data structures.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class ResourceType(str, Enum):
    EC2 = "ec2_instance"
    RDS = "rds_database"
    EBS = "ebs_volume"
    ALB = "alb_load_balancer"


class ResourceStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    AVAILABLE = "available"       # EBS: unattached
    IN_USE = "in-use"            # EBS: attached
    REBOOTING = "rebooting"
    TERMINATED = "terminated"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionCommand(str, Enum):
    TERMINATE = "terminate"
    SCALE = "scale"
    REBOOT = "reboot"
    INSPECT = "inspect"
    WAIT = "wait"


# ─── Data Models ──────────────────────────────────────────────────────────────

class Resource(BaseModel):
    """Represents a single cloud resource (EC2, RDS, EBS, ALB)."""
    id: str
    name: str = ""
    type: ResourceType
    status: ResourceStatus
    instance_size: str = ""              # e.g., "t3.micro", "db.t3.medium"
    cpu_utilization: float = 0.0         # 0.0 – 100.0
    memory_utilization: float = 0.0      # 0.0 – 100.0
    cost_per_hour: float = 0.0
    attached_to: Optional[str] = None    # For EBS: the EC2 id it's attached to
    tags: Dict[str, str] = Field(default_factory=dict)


class Alert(BaseModel):
    """Represents an active monitoring alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    resource_id: Optional[str] = None
    metric_name: Optional[str] = None    # e.g., "Latency", "CPUUtilization"
    metric_value: Optional[float] = None


class Observation(BaseModel):
    """
    The full observation returned by state() and step().
    Contains everything the agent can see about the infrastructure.
    """
    resources: List[Resource] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)
    total_hourly_cost: float = 0.0
    system_uptime: float = 100.0         # 0.0 – 100.0
    step_number: int = 0
    max_steps: int = 15
    budget_limit: Optional[float] = None # Some tasks have a budget constraint


class Action(BaseModel):
    """
    An action the agent submits to step().
    """
    command: ActionCommand
    resource_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    # params examples:
    #   scale: {"target_size": "db.t3.medium"}
    #   inspect: {} (just returns detailed info about the resource)


class StepResult(BaseModel):
    """The result of calling step(action)."""
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
