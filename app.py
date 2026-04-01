"""
Cloud SRE & FinOps Simulator — Hackathon Dashboard

A visually stunning OpenEnv-compliant application featuring:
- Dark glassmorphism dashboard with metric cards
- Interactive infrastructure topology visualization
- Real-time cost optimization charts (Plotly)
- ReAct AI agent with visible reasoning traces
- Live Agent Arena leaderboard
- Animated alerts with severity coloring

Exposes REST API + Gradio dashboard, deployed to HF Spaces on port 7860.
"""

import json
import os
import time
import random
import threading
import gradio as gr
import plotly.graph_objects as go
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from env import CloudSREEnv
from models import Action, ActionCommand, Observation, StepResult
from tasks import list_tasks

# ─── FastAPI Application ──────────────────────────────────────────────────────

api = FastAPI(
    title="Cloud SRE OpenEnv — AI Agent Dashboard",
    version="2.0.0",
    description="OpenEnv-compliant Cloud SRE & FinOps simulator with ReAct reasoning and multi-model benchmarking.",
)

# ─── Global State ─────────────────────────────────────────────────────────────

env = CloudSREEnv(max_steps=15)
current_task = None
agent_trace_md = ""
arena_html_cache = ""

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

# ─── Helper ───────────────────────────────────────────────────────────────────

def serialize_obs(obs: Observation) -> Dict[str, Any]:
    return obs.model_dump()

# ─── REST API Endpoints ──────────────────────────────────────────────────────

@api.post("/reset", response_model=ObservationResponse, tags=["Environment Control"])
async def reset_endpoint(request: ResetRequest = Body(default_factory=ResetRequest)):
    global current_task
    current_task = request.task_id
    obs = env.reset(request.task_id, seed=request.seed)
    return ObservationResponse(observation=serialize_obs(obs), reward=None, done=False,
                               info={"task_id": request.task_id})

@api.post("/step", response_model=ObservationResponse, tags=["Environment Control"])
async def step_endpoint(request: StepRequest):
    action_data = request.action
    try:
        command = ActionCommand(action_data.get("command", ""))
    except ValueError:
        return JSONResponse(status_code=422,
            content={"detail": f"Invalid command: {action_data.get('command')}. "
                     f"Valid: {[c.value for c in ActionCommand]}"})
    action = Action(command=command, resource_id=action_data.get("resource_id"),
                    params=action_data.get("params", {}))
    result: StepResult = env.step(action)
    return ObservationResponse(observation=serialize_obs(result.observation),
                               reward=result.reward, done=result.done, info=result.info)

@api.get("/state", response_model=StateResponse, tags=["State Management"])
async def state_endpoint():
    obs = env.state()
    return StateResponse(episode_id=None, step_count=obs.step_number,
                         is_done=env._done, task_id=current_task)

@api.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_endpoint():
    return HealthResponse(status="healthy")

@api.get("/schema", response_model=SchemaResponse, tags=["Schema"])
async def schema_endpoint():
    return SchemaResponse(action=Action.model_json_schema(),
                          observation=Observation.model_json_schema(),
                          state=StateResponse.model_json_schema())

@api.get("/tasks", tags=["Environment Info"])
async def tasks_endpoint():
    return list_tasks()

@api.get("/metadata", tags=["Environment Info"])
async def metadata_endpoint():
    return {
        "name": "cloud-sre-simulator", "version": "2.0.0",
        "description": "OpenEnv environment with ReAct AI agents and multi-model benchmarking.",
        "tasks": list_tasks(),
        "action_space": {"type": "discrete", "commands": [c.value for c in ActionCommand]},
        "reward_range": [-1.0, 1.0], "max_steps": 15,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global Dark Theme ── */
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
}
.gradio-container { max-width: 1400px !important; }

/* ── Dashboard Header ── */
.dash-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.dash-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #4ade80, #22d3ee, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.dash-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
    margin: 0;
}
.dash-badges {
    margin-top: 0.6rem;
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
}
.dash-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.metric-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.val-green { color: #4ade80; }
.val-red { color: #f87171; }
.val-yellow { color: #fbbf24; }
.val-blue { color: #60a5fa; }

/* ── Topology ── */
.topo-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.topo-title {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.8rem;
}
.topo-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.topo-node {
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.15s;
    min-width: 140px;
}
.topo-node:hover { transform: scale(1.05); }
.topo-node .node-id { font-weight: 700; margin-bottom: 2px; }
.topo-node .node-meta { font-size: 0.7rem; opacity: 0.8; }
.node-healthy { background: rgba(74,222,128,0.12); color: #4ade80; border-color: rgba(74,222,128,0.3); }
.node-degraded { background: rgba(251,191,36,0.12); color: #fbbf24; border-color: rgba(251,191,36,0.3); }
.node-critical { background: rgba(248,113,113,0.12); color: #f87171; border-color: rgba(248,113,113,0.3); }
.node-stopped { background: rgba(156,163,175,0.12); color: #9ca3af; border-color: rgba(156,163,175,0.3); }

/* ── Alerts ── */
.alerts-panel { display: flex; flex-direction: column; gap: 0.5rem; }
.alert-item {
    padding: 0.7rem 1rem;
    border-radius: 10px;
    font-size: 0.82rem;
    border-left: 4px solid;
    animation: alertSlide 0.4s ease-out;
}
@keyframes alertSlide {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}
.alert-crit { background: rgba(239,68,68,0.1); border-color: #ef4444; color: #fca5a5; }
.alert-warn { background: rgba(245,158,11,0.1); border-color: #f59e0b; color: #fde68a; }
.alert-info { background: rgba(59,130,246,0.1); border-color: #3b82f6; color: #93c5fd; }
.alert-sev { font-weight: 700; margin-right: 0.4rem; }

/* ── Arena Leaderboard ── */
.arena-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    overflow: hidden;
}
.arena-header h3 {
    color: #e2e8f0;
    margin: 0 0 0.2rem 0;
    font-size: 1.1rem;
}
.arena-task {
    color: rgba(255,255,255,0.4);
    font-size: 0.78rem;
    margin-bottom: 1rem;
}
.lb-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.03);
    animation: lbSlide 0.5s ease-out both;
}
@keyframes lbSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.lb-rank { font-size: 1.5rem; min-width: 2.5rem; text-align: center; }
.lb-info { flex: 1; }
.lb-name { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }
.lb-bar-container {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.lb-bar {
    height: 100%;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    border-radius: 3px;
    transition: width 1s ease-out;
}
.lb-stats { text-align: right; min-width: 140px; }
.lb-score { font-size: 1.4rem; font-weight: 800; color: #4ade80; }
.lb-meta { font-size: 0.72rem; color: rgba(255,255,255,0.4); }
.mock-tag { font-size: 0.65rem; color: rgba(255,255,255,0.3); font-weight: 400; }
.live-tag { font-size: 0.6rem; color: #4ade80; font-weight: 700; background: rgba(74,222,128,0.15);
            padding: 1px 6px; border-radius: 8px; }
.q-badge { font-size: 0.6rem; padding: 1px 6px; border-radius: 8px; font-weight: 600; }
.q-high { background: rgba(74,222,128,0.15); color: #4ade80; }
.q-med { background: rgba(251,191,36,0.15); color: #fbbf24; }
.q-low { background: rgba(248,113,113,0.15); color: #f87171; }
.q-err { background: rgba(156,163,175,0.15); color: #9ca3af; }
.arena-empty { color: rgba(255,255,255,0.4); text-align: center; padding: 2rem; }
"""

# ─── Dashboard Helper Functions ──────────────────────────────────────────────

def generate_metric_cards(obs) -> str:
    """Generate the 4 metric cards."""
    cost = obs.total_hourly_cost
    uptime = obs.system_uptime
    step = obs.step_number
    max_s = obs.max_steps
    n_alerts = len(obs.alerts)

    cost_color = "val-green" if cost < 1.0 else ("val-yellow" if cost < 5.0 else "val-red")
    uptime_color = "val-green" if uptime >= 95 else ("val-yellow" if uptime >= 70 else "val-red")
    alert_color = "val-green" if n_alerts == 0 else ("val-yellow" if n_alerts <= 2 else "val-red")

    return f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.2rem;">
        <div style="background:linear-gradient(135deg,rgba(74,222,128,0.08),rgba(74,222,128,0.02)); border:1px solid rgba(74,222,128,0.2); border-radius:14px; padding:1.2rem; text-align:center;">
            <div style="font-size:1.5rem;">💰</div>
            <div style="font-size:1.8rem; font-weight:700; color:{('#4ade80' if cost < 1.0 else ('#fbbf24' if cost < 5.0 else '#f87171'))};">${cost:.2f}</div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.5px;">Hourly Cost</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(96,165,250,0.08),rgba(96,165,250,0.02)); border:1px solid rgba(96,165,250,0.2); border-radius:14px; padding:1.2rem; text-align:center;">
            <div style="font-size:1.5rem;">📈</div>
            <div style="font-size:1.8rem; font-weight:700; color:{('#4ade80' if uptime >= 95 else ('#fbbf24' if uptime >= 70 else '#f87171'))};">{uptime:.0f}%</div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.5px;">System Uptime</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(167,139,250,0.08),rgba(167,139,250,0.02)); border:1px solid rgba(167,139,250,0.2); border-radius:14px; padding:1.2rem; text-align:center;">
            <div style="font-size:1.5rem;">⚡</div>
            <div style="font-size:1.8rem; font-weight:700; color:#60a5fa;">{step}/{max_s}</div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.5px;">Step Progress</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(248,113,113,0.08),rgba(248,113,113,0.02)); border:1px solid rgba(248,113,113,0.2); border-radius:14px; padding:1.2rem; text-align:center;">
            <div style="font-size:1.5rem;">🚨</div>
            <div style="font-size:1.8rem; font-weight:700; color:{('#4ade80' if n_alerts == 0 else ('#fbbf24' if n_alerts <= 2 else '#f87171'))};">{n_alerts}</div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.5px;">Active Alerts</div>
        </div>
    </div>
    """


def generate_topology_html(obs) -> str:
    """Generate infrastructure topology visualization."""
    if not obs.resources:
        return '<div class="topo-container"><div class="topo-title">Infrastructure Topology</div><p style="color:rgba(255,255,255,0.4)">No resources loaded.</p></div>'

    alert_rids = {a.resource_id for a in obs.alerts if a.severity.value == "critical"}

    icons = {
        "ec2_instance": "🖥",
        "rds_database": "🗄",
        "ebs_volume": "💾",
        "alb_load_balancer": "⚖",
    }

    nodes = []
    color_map = {
        "critical": ("248,113,113", "#f87171"),
        "stopped": ("156,163,175", "#9ca3af"),
        "degraded": ("251,191,36", "#fbbf24"),
        "healthy": ("74,222,128", "#4ade80"),
    }

    for r in obs.resources:
        if r.id in alert_rids:
            state = "critical"
        elif r.status.value == "stopped":
            state = "stopped"
        elif r.cpu_utilization > 80:
            state = "degraded"
        else:
            state = "healthy"

        rgb, text_color = color_map[state]
        icon = icons.get(r.type.value, "📦")
        env_tag = r.tags.get("env", "")
        env_badge = f' <span style="opacity:0.6">[{env_tag}]</span>' if env_tag else ""

        nodes.append(
            f'<div style="padding:0.6rem 1rem;border-radius:10px;font-size:0.82rem;font-weight:600;'
            f'border:1px solid rgba({rgb},0.3);background:rgba({rgb},0.12);color:{text_color};'
            f'min-width:140px;">'
            f'<div style="font-weight:700;margin-bottom:2px;">{icon} {r.id}{env_badge}</div>'
            f'<div style="font-size:0.7rem;opacity:0.8;">{r.status.value} &middot; CPU {r.cpu_utilization:.0f}% &middot; ${r.cost_per_hour:.3f}/hr</div>'
            f'</div>'
        )

    joined = "".join(nodes)
    return (
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1.2rem;margin-bottom:1rem;">'
        f'<div style="font-size:0.85rem;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">🌐 Infrastructure Topology &middot; {len(obs.resources)} Resources</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem;">{joined}</div>'
        f'</div>'
    )


def generate_alerts_html(obs) -> str:
    """Generate styled alerts panel."""
    if not obs.alerts:
        return '<div style="display:flex;flex-direction:column;gap:0.5rem;"><div style="padding:0.7rem 1rem;border-radius:10px;font-size:0.82rem;border-left:4px solid #3b82f6;background:rgba(59,130,246,0.1);color:#93c5fd;">✅ No active alerts. All systems nominal.</div></div>'

    color_map = {
        "critical": ("#ef4444", "rgba(239,68,68,0.1)", "#fca5a5"),
        "warning": ("#f59e0b", "rgba(245,158,11,0.1)", "#fde68a"),
        "info": ("#3b82f6", "rgba(59,130,246,0.1)", "#93c5fd"),
    }
    items = []
    for a in obs.alerts:
        border_c, bg_c, text_c = color_map.get(a.severity.value, color_map["info"])
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(a.severity.value, "⚪")
        rid_text = f" &middot; <code>{a.resource_id}</code>" if a.resource_id else ""
        items.append(
            f'<div style="padding:0.7rem 1rem;border-radius:10px;font-size:0.82rem;border-left:4px solid {border_c};background:{bg_c};color:{text_c};">'
            f'<span style="font-weight:700;margin-right:0.4rem;">{icon} [{a.severity.value.upper()}]</span>'
            f'{a.message}{rid_text}'
            f'</div>'
        )

    joined = "".join(items)

    return f'<div style="display:flex;flex-direction:column;gap:0.5rem;">{joined}</div>'


def generate_cost_chart(cost_history: List[float]) -> go.Figure:
    """Generate a Plotly cost-over-time chart."""
    fig = go.Figure()

    # Cost line
    fig.add_trace(go.Scatter(
        x=list(range(len(cost_history))),
        y=cost_history,
        mode='lines+markers',
        name='Hourly Cost ($/hr)',
        line=dict(color='#4ade80', width=3, shape='spline'),
        marker=dict(size=8, color='#4ade80',
                    line=dict(width=2, color='rgba(74,222,128,0.3)')),
        fill='tozeroy',
        fillcolor='rgba(74,222,128,0.08)',
    ))

    # Budget line if available
    budget = env._state.get("budget_limit")
    if budget:
        fig.add_hline(y=budget, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Budget: ${budget:.2f}",
                      annotation_font_color="#f87171")

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Step',
        yaxis_title='Cost ($/hr)',
        title=dict(text='💰 Cost Optimization Over Time', font=dict(size=14)),
        font=dict(color='rgba(255,255,255,0.7)', size=11),
        height=280,
        margin=dict(l=50, r=20, t=45, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', dtick=1),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        showlegend=False,
    )
    return fig


# ─── Gradio Event Handlers ───────────────────────────────────────────────────

def reset_task(task_id: str, use_seed: bool):
    """Reset environment and return all dashboard components."""
    global current_task, agent_trace_md
    current_task = task_id
    agent_trace_md = ""

    seed = random.randint(1, 99999) if use_seed else None
    obs = env.reset(task_id, seed=seed)
    desc = env.get_task_description()

    seed_info = f" (Seed: {seed})" if seed else " (Fixed scenario)"
    task_md = f"## 🎯 Task: {task_id}{seed_info}\n{desc}"

    metrics = generate_metric_cards(obs)
    topology = generate_topology_html(obs)
    alerts = generate_alerts_html(obs)
    cost_chart = generate_cost_chart(env.get_cost_history())

    return task_md, metrics, topology, cost_chart, alerts, "", ""


def execute_action(command: str, resource_id: str, target_size: str):
    """Execute a manual action and update dashboard."""
    if current_task is None:
        return "", "", generate_cost_chart([0]), "", "⚠️ Select a task first!"

    try:
        cmd = ActionCommand(command)
    except ValueError:
        return "", "", generate_cost_chart([0]), "", "⚠️ Invalid command!"

    params = {}
    if target_size and target_size.strip():
        params["target_size"] = target_size.strip()

    action = Action(command=cmd, resource_id=resource_id.strip() if resource_id else None, params=params)
    result = env.step(action)
    obs = result.observation

    metrics = generate_metric_cards(obs)
    topology = generate_topology_html(obs)
    cost_chart = generate_cost_chart(env.get_cost_history())
    alerts = generate_alerts_html(obs)

    # Build result markdown
    info = result.info
    parts = [
        f"**Step {info.get('step', '?')}** — `{command}({resource_id or ''})`",
        f"Reward: `{result.reward:+.4f}` | Cumulative: `{info.get('cumulative_reward', 0):+.4f}`",
    ]
    if info.get("action_result"):
        parts.append(f"```\n{info['action_result']}\n```")
    if info.get("chaos_event"):
        parts.append(f"\n{info['chaos_event']}")
    if result.done:
        final = info.get("final_score", 0)
        breakdown = info.get("grading_breakdown", {})
        parts.append(f"\n## 🏁 Episode Complete!")
        parts.append(f"**Final Score: {final:.2f} / 1.00**")
        parts.append(f"```json\n{json.dumps(breakdown, indent=2)}\n```")

    result_md = "\n".join(parts)
    return metrics, topology, cost_chart, alerts, result_md


def run_ai_agent(task_id: str, use_seed: bool, api_key: str):
    """Run the ReAct AI agent and stream reasoning traces."""
    global agent_trace_md, current_task

    if not api_key and not os.environ.get("OPENAI_API_KEY"):
        return ("", "", generate_cost_chart([0]), "",
                "⚠️ Please provide an OpenAI API key.",
                "⚠️ No API key provided.", "")

    key = api_key.strip() if api_key else os.environ.get("OPENAI_API_KEY", "")
    current_task = task_id
    seed = random.randint(1, 99999) if use_seed else None

    from react_agent import ReActAgent
    agent = ReActAgent(api_key=key, model="gpt-4o-mini")
    trace = agent.run_episode(env, task_id, seed=seed)

    agent_trace_md = trace.to_markdown()
    obs = env.state()

    metrics = generate_metric_cards(obs)
    topology = generate_topology_html(obs)
    cost_chart = generate_cost_chart(env.get_cost_history())
    alerts = generate_alerts_html(obs)

    # Final result
    parts = [
        f"## 🏁 AI Agent Complete!",
        f"**Model:** {trace.model} | **Steps:** {len(trace.steps)} | **Time:** {trace.total_time:.1f}s",
        f"**Final Score: {trace.final_score:.2f} / 1.00**",
        f"**Cost:** ${trace.initial_cost:.4f}/hr → ${trace.final_cost:.4f}/hr",
    ]
    if trace.grading_breakdown:
        parts.append(f"```json\n{json.dumps(trace.grading_breakdown, indent=2)}\n```")

    result_md = "\n".join(parts)
    return metrics, topology, cost_chart, alerts, result_md, agent_trace_md, ""


def run_arena(task_id: str, api_key: str):
    """Run the Agent Arena comparison."""
    global arena_html_cache

    key = api_key.strip() if api_key else os.environ.get("OPENAI_API_KEY", "")

    from arena import AgentArena
    arena = AgentArena(api_key=key if key else None)

    seed = 42  # Fixed seed for fair comparison
    results = arena.run_arena(task_id, seed=seed, models=["gpt-4o-mini"])
    arena_html_cache = arena.generate_leaderboard_html()

    return arena_html_cache


# ─── Build Gradio Dashboard ──────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        block_background_fill="rgba(30, 41, 59, 0.8)",
        block_background_fill_dark="rgba(30, 41, 59, 0.8)",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_color_dark="rgba(255,255,255,0.08)",
        input_background_fill="rgba(15, 23, 42, 0.9)",
        input_background_fill_dark="rgba(15, 23, 42, 0.9)",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
    ),
    css=CUSTOM_CSS,
    title="Cloud SRE & FinOps Simulator — AI Agent Dashboard"
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="dash-header">
        <h1>🏗 Cloud SRE & FinOps Simulator</h1>
        <p class="dash-subtitle">OpenEnv-Compliant AI Agent Environment with ReAct Reasoning</p>
        <div class="dash-badges">
            <span class="dash-badge">🧠 ReAct Agent</span>
            <span class="dash-badge">🔀 Procedural Generation</span>
            <span class="dash-badge">⚡ Chaos Injection</span>
            <span class="dash-badge">🏟 Agent Arena</span>
            <span class="dash-badge">📊 FinOps Analytics</span>
        </div>
    </div>
    """)

    # ── Controls Row ──
    with gr.Row():
        with gr.Column(scale=2):
            task_dropdown = gr.Dropdown(
                choices=[t["id"] for t in list_tasks()],
                label="🎯 Select Task",
                value="phantom_volume_cleanup",
            )
        with gr.Column(scale=1):
            seed_toggle = gr.Checkbox(label="🔀 Randomize (Procedural)", value=True)
        with gr.Column(scale=1):
            reset_btn = gr.Button("🔄 Reset Task", variant="primary", size="lg")
        with gr.Column(scale=1):
            ai_btn = gr.Button("🤖 Auto-Solve (AI)", variant="secondary", size="lg")

    api_key_input = gr.Textbox(
        label="🔑 OpenAI API Key",
        placeholder="sk-... (or set OPENAI_API_KEY env var)",
        type="password",
        visible=True,
    )

    task_desc = gr.Markdown("Select a task and click **Reset** to begin.")

    # ── Metric Cards ──
    metrics_html = gr.HTML("")

    # ── Topology ──
    topology_html = gr.HTML("")

    # ── Charts + Reasoning Row ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            cost_plot = gr.Plot(label="Cost Over Time")
        with gr.Column(scale=1):
            reasoning_panel = gr.Markdown(
                "### 🧠 Agent Reasoning Trace\nRun the AI agent to see its Think → Act → Observe process.",
                label="Agent Reasoning",
            )

    # ── Alerts ──
    alerts_html = gr.HTML('<div class="alerts-panel"><div class="alert-item alert-info">Select a task to begin.</div></div>')

    # ── Manual Action Controls ──
    with gr.Accordion("⚙️ Manual Actions", open=False):
        with gr.Row():
            cmd_dropdown = gr.Dropdown(
                choices=["inspect", "terminate", "scale", "reboot", "wait"],
                label="Command", value="inspect",
            )
            rid_input = gr.Textbox(label="Resource ID", placeholder="e.g., ec2-web-001")
            size_input = gr.Textbox(label="Target Size (for scale)", placeholder="e.g., db.t3.medium")
            action_btn = gr.Button("▶ Execute", variant="secondary")

    result_box = gr.Markdown("")

    # ── Arena Section ──
    with gr.Accordion("🏟 Agent Arena — Multi-Model Comparison", open=False):
        gr.Markdown("Compare AI models on the **same seeded scenario**. GPT models run live; Claude & Llama show simulated results.")
        arena_btn = gr.Button("🚀 Run Arena Comparison", variant="primary")
        arena_output = gr.HTML('<div class="arena-empty">Click "Run Arena Comparison" to start.</div>')

    # ── Wire Events ──
    reset_btn.click(
        fn=reset_task,
        inputs=[task_dropdown, seed_toggle],
        outputs=[task_desc, metrics_html, topology_html, cost_plot, alerts_html, result_box, reasoning_panel],
    )

    action_btn.click(
        fn=execute_action,
        inputs=[cmd_dropdown, rid_input, size_input],
        outputs=[metrics_html, topology_html, cost_plot, alerts_html, result_box],
    )

    ai_btn.click(
        fn=run_ai_agent,
        inputs=[task_dropdown, seed_toggle, api_key_input],
        outputs=[metrics_html, topology_html, cost_plot, alerts_html, result_box, reasoning_panel, api_key_input],
    )

    arena_btn.click(
        fn=run_arena,
        inputs=[task_dropdown, api_key_input],
        outputs=[arena_output],
    )

# ── Mount Gradio at root so HF Spaces iframe works ──
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
