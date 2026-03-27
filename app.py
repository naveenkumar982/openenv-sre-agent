"""
Gradio-based UI for the Cloud SRE OpenEnv.
Deploys to Hugging Face Spaces.
"""

import gradio as gr
import json
from env import CloudSREEnv  # type: ignore
from models import Action, ActionCommand  # type: ignore
from tasks import list_tasks  # type: ignore

# ─── Global Environment Instance ──────────────────────────────────────────────

env = CloudSREEnv(max_steps=15)
current_task = None


def format_resources(obs) -> str:
    """Format resources into a readable table."""
    if not obs.resources:
        return "No resources."

    lines = []
    lines.append(f"{'ID':<28} {'Type':<16} {'Status':<12} {'Size':<14} {'CPU%':>6} {'$/hr':>8} {'Tags'}")
    lines.append("─" * 110)

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
        return "✅ No active alerts."

    lines = []
    for a in obs.alerts:
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(a.severity.value, "⚪")
        lines.append(f"{icon} [{a.severity.value.upper()}] {a.message}")
        if a.resource_id:
            lines.append(f"   └─ Resource: {a.resource_id}")

    return "\n".join(lines)


def format_state(obs) -> str:
    """Format the full state display."""
    header = (
        f"💰 Total Cost: ${obs.total_hourly_cost:.4f}/hr  |  "
        f"📊 Uptime: {obs.system_uptime:.1f}%  |  "
        f"🔄 Step: {obs.step_number}/{obs.max_steps}"
    )
    if obs.budget_limit:
        header += f"  |  🎯 Budget: ${obs.budget_limit:.2f}/hr"

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
        f"## 🎯 Task: {task_id}\n{desc}",
        state_header,
        resources_text,
        alerts_text,
        "",  # Clear action log
        "",  # Clear result
    )


def execute_action(command: str, resource_id: str, target_size: str):
    """Execute an action in the environment."""
    if current_task is None:
        return "⚠️ Select a task first!", "", "", "", ""

    try:
        cmd = ActionCommand(command)
    except ValueError:
        return "⚠️ Invalid command!", "", "", "", ""

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

    # Build result message
    info = result.info
    result_parts = [
        f"**Step {info.get('step', '?')}** — `{command}({resource_id or ''})`",
        f"Reward: `{result.reward:+.4f}` | Cumulative: `{info.get('cumulative_reward', 0):+.4f}`",
    ]

    if "action_result" in info:
        result_parts.append(f"```\n{info['action_result']}\n```")

    if result.done:
        final = info.get("final_score", 0)
        breakdown = info.get("grading_breakdown", {})
        result_parts.append(f"\n## 🏁 Episode Complete!")
        result_parts.append(f"**Final Score: {final:.2f} / 1.00**")
        result_parts.append(f"```json\n{json.dumps(breakdown, indent=2)}\n```")

    result_text = "\n".join(result_parts)

    return state_header, resources_text, alerts_text, "", result_text


# ─── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ☁️ Cloud SRE & FinOps Simulator
        ### An OpenEnv AI Agent Environment

        Manage a simulated cloud infrastructure. Diagnose outages, optimize costs,
        and restore services — just like a real Site Reliability Engineer.
        """
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[t["id"] for t in list_tasks()],
            label="Select Task",
            value="phantom_volume_cleanup",
        )
        reset_btn = gr.Button("🔄 Reset Task", variant="primary")

    task_desc = gr.Markdown("Select a task and click Reset to begin.")
    state_header = gr.Textbox(label="📊 System Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=2):
            resources_box = gr.Textbox(
                label="🖥️ Resources",
                lines=12,
                interactive=False,
            )
        with gr.Column(scale=1):
            alerts_box = gr.Textbox(
                label="🚨 Alerts",
                lines=12,
                interactive=False,
            )

    gr.Markdown("---")
    gr.Markdown("### 🎮 Execute Action")

    with gr.Row():
        cmd_dropdown = gr.Dropdown(
            choices=["inspect", "terminate", "scale", "reboot", "wait"],
            label="Command",
            value="inspect",
        )
        rid_input = gr.Textbox(label="Resource ID", placeholder="e.g., ec2-web-001")
        size_input = gr.Textbox(label="Target Size (for scale)", placeholder="e.g., db.t3.medium")
        action_btn = gr.Button("▶️ Execute", variant="secondary")

    result_box = gr.Markdown("Results will appear here after each action.")

    # ── Wire up events ──
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


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
