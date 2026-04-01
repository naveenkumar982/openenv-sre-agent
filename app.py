"""
FastAPI + Gradio application for the Cloud SRE OpenEnv.

Exposes the OpenEnv-compliant REST API endpoints:
  POST /reset   — Reset environment to initial state
  POST /step    — Execute an action
  GET  /state   — Get current observation
  GET  /health  — Health check
  GET  /schema  — JSON schemas for action/observation

Also mounts the Gradio interactive UI at /gradio.

Deploys to Hugging Face Spaces on port 7860.
"""

import json
import gradio as gr
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from env import CloudSREEnv
from models import Action, ActionCommand, Observation, StepResult
from tasks import list_tasks

# ─── FastAPI Application ──────────────────────────────────────────────────────

api = FastAPI(
    title="Cloud SRE OpenEnv — HTTP API",
    version="1.0.0",
    description=(
        "An OpenEnv-compliant environment simulating Cloud SRE & FinOps operations. "
        "Manage cloud infrastructure: diagnose outages, optimize costs, and restore services."
    ),
)

# ─── Global Environment Instance ──────────────────────────────────────────────

env = CloudSREEnv(max_steps=15)
current_task = None


# ─── Request / Response Models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "phantom_volume_cleanup"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ObservationResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "healthy"


class SchemaResponse(BaseModel):
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state: Dict[str, Any]


class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    is_done: bool = False
    task_id: Optional[str] = None


# ─── Helper: Serialize Observation ────────────────────────────────────────────

def serialize_obs(obs: Observation) -> Dict[str, Any]:
    """Convert an Observation to a JSON-serializable dict."""
    return obs.model_dump()


# ─── OpenEnv REST API Endpoints ──────────────────────────────────────────────

@api.post("/reset", response_model=ObservationResponse, tags=["Environment Control"])
async def reset_endpoint(request: ResetRequest = Body(default_factory=ResetRequest)):
    """
    Reset the environment to a specific task's initial state.
    Returns the initial observation.
    """
    global current_task
    current_task = request.task_id
    obs = env.reset(request.task_id)
    return ObservationResponse(
        observation=serialize_obs(obs),
        reward=None,
        done=False,
        info={"task_id": request.task_id},
    )


@api.post("/step", response_model=ObservationResponse, tags=["Environment Control"])
async def step_endpoint(request: StepRequest):
    """
    Execute an action in the environment.
    Returns the resulting observation, reward, done, and info.
    """
    action_data = request.action

    # Parse the action
    try:
        command = ActionCommand(action_data.get("command", ""))
    except ValueError:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Invalid command: {action_data.get('command')}. "
                     f"Valid: {[c.value for c in ActionCommand]}"},
        )

    action = Action(
        command=command,
        resource_id=action_data.get("resource_id"),
        params=action_data.get("params", {}),
    )

    result: StepResult = env.step(action)
    return ObservationResponse(
        observation=serialize_obs(result.observation),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@api.get("/state", response_model=StateResponse, tags=["State Management"])
async def state_endpoint():
    """Get the current environment state metadata."""
    obs = env.state()
    return StateResponse(
        episode_id=None,
        step_count=obs.step_number,
        is_done=env._done,
        task_id=current_task,
    )


@api.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_endpoint():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@api.get("/schema", response_model=SchemaResponse, tags=["Schema"])
async def schema_endpoint():
    """Get JSON schemas for action, observation, and state."""
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=StateResponse.model_json_schema(),
    )


@api.get("/tasks", tags=["Environment Info"])
async def tasks_endpoint():
    """List all available tasks."""
    return list_tasks()


@api.get("/metadata", tags=["Environment Info"])
async def metadata_endpoint():
    """Get environment metadata."""
    return {
        "name": "cloud-sre-simulator",
        "version": "1.0.0",
        "description": (
            "An OpenEnv environment simulating Cloud SRE & FinOps operations. "
            "An AI agent manages a simulated cloud infrastructure."
        ),
        "tasks": list_tasks(),
        "action_space": {"type": "discrete", "commands": [c.value for c in ActionCommand]},
        "reward_range": [-1.0, 1.0],
        "max_steps": 15,
    }


# ─── Gradio Interface (mounted as sub-app) ───────────────────────────────────

def format_resources(obs) -> str:
    """Format resources into a readable table."""
    if not obs.resources:
        return "No resources."
    lines = []
    lines.append(f"{'ID':<28} {'Type':<16} {'Status':<12} {'Size':<14} {'CPU%':>6} {'$/hr':>8} {'Tags'}")
    lines.append("\u2500" * 110)
    for r in obs.resources:
        tags_str = ", ".join(f"{k}={v}" for k, v in r.tags.items()) if r.tags else ""
        line = (
            f"{r.id:<28} {r.type.value:<16} {r.status.value:<12} "
            f"{r.instance_size:<14} {r.cpu_utilization:>5.1f}% "
            f"${r.cost_per_hour:>7.4f} {tags_str}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_alerts(obs) -> str:
    """Format alerts into a readable list."""
    if not obs.alerts:
        return "No active alerts."
    lines = []
    for a in obs.alerts:
        icon = {"critical": "[CRIT]", "warning": "[WARN]", "info": "[INFO]"}.get(a.severity.value, "[?]")
        lines.append(f"{icon} [{a.severity.value.upper()}] {a.message}")
        if a.resource_id:
            lines.append(f"   Resource: {a.resource_id}")
    return "\n".join(lines)


def format_state(obs) -> str:
    """Format the full state display."""
    header = (
        f"Total Cost: ${obs.total_hourly_cost:.4f}/hr  |  "
        f"Uptime: {obs.system_uptime:.1f}%  |  "
        f"Step: {obs.step_number}/{obs.max_steps}"
    )
    if obs.budget_limit:
        header += f"  |  Budget: ${obs.budget_limit:.2f}/hr"
    return header


def reset_task(task_id: str):
    """Reset the environment to a selected task."""
    global current_task
    current_task = task_id
    obs = env.reset(task_id)
    desc = env.get_task_description()
    state_header = format_state(obs)
    resources_text = format_resources(obs)
    alerts_text = format_alerts(obs)
    return (
        f"## Task: {task_id}\n{desc}",
        state_header,
        resources_text,
        alerts_text,
        "",
        "",
    )


def execute_action(command: str, resource_id: str, target_size: str):
    """Execute an action in the environment."""
    if current_task is None:
        return "Select a task first!", "", "", "", ""
    try:
        cmd = ActionCommand(command)
    except ValueError:
        return "Invalid command!", "", "", "", ""
    params = {}
    if target_size and target_size.strip():
        params["target_size"] = target_size.strip()
    action = Action(
        command=cmd,
        resource_id=resource_id.strip() if resource_id else None,
        params=params,
    )
    result = env.step(action)
    obs = result.observation
    state_header = format_state(obs)
    resources_text = format_resources(obs)
    alerts_text = format_alerts(obs)
    info = result.info
    result_parts = [
        f"**Step {info.get('step', '?')}** -- `{command}({resource_id or ''})`",
        f"Reward: `{result.reward:+.4f}` | Cumulative: `{info.get('cumulative_reward', 0):+.4f}`",
    ]
    if "action_result" in info:
        result_parts.append(f"```\n{info['action_result']}\n```")
    if result.done:
        final = info.get("final_score", 0)
        breakdown = info.get("grading_breakdown", {})
        result_parts.append(f"\n## Episode Complete!")
        result_parts.append(f"**Final Score: {final:.2f} / 1.00**")
        result_parts.append(f"```json\n{json.dumps(breakdown, indent=2)}\n```")
    result_text = "\n".join(result_parts)
    return state_header, resources_text, alerts_text, "", result_text


with gr.Blocks(title="Cloud SRE & FinOps Simulator") as demo:
    gr.Markdown(
        """
        # Cloud SRE & FinOps Simulator
        ### An OpenEnv AI Agent Environment

        Manage a simulated cloud infrastructure. Diagnose outages, optimize costs,
        and restore services -- just like a real Site Reliability Engineer.
        """
    )
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[t["id"] for t in list_tasks()],
            label="Select Task",
            value="phantom_volume_cleanup",
        )
        reset_btn = gr.Button("Reset Task", variant="primary")
    task_desc = gr.Markdown("Select a task and click Reset to begin.")
    state_header = gr.Textbox(label="System Status", interactive=False)
    with gr.Row():
        with gr.Column(scale=2):
            resources_box = gr.Textbox(label="Resources", lines=12, interactive=False)
        with gr.Column(scale=1):
            alerts_box = gr.Textbox(label="Alerts", lines=12, interactive=False)
    gr.Markdown("---")
    gr.Markdown("### Execute Action")
    with gr.Row():
        cmd_dropdown = gr.Dropdown(
            choices=["inspect", "terminate", "scale", "reboot", "wait"],
            label="Command",
            value="inspect",
        )
        rid_input = gr.Textbox(label="Resource ID", placeholder="e.g., ec2-web-001")
        size_input = gr.Textbox(label="Target Size (for scale)", placeholder="e.g., db.t3.medium")
        action_btn = gr.Button("Execute", variant="secondary")
    result_box = gr.Markdown("Results will appear here after each action.")
    reset_btn.click(
        fn=reset_task,
        inputs=[task_dropdown],
        outputs=[task_desc, state_header, resources_box, alerts_box, rid_input, result_box],
    )
    action_btn.click(
        fn=execute_action,
        inputs=[cmd_dropdown, rid_input, size_input],
        outputs=[state_header, resources_box, alerts_box, rid_input, result_box],
    )

# Mount Gradio inside the FastAPI app
app = gr.mount_gradio_app(api, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
