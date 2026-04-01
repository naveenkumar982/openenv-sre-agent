"""
Baseline inference script for the Cloud SRE OpenEnv.
Uses the OpenAI API (GPT-4o) to run an LLM agent through all 3 tasks.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline.py
"""

import os
import json
import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from openai import OpenAI
from env import CloudSREEnv  # type: ignore
from models import Action, ActionCommand  # type: ignore
from tasks import list_tasks  # type: ignore


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL = "gpt-4o"
NUM_RUNS = 3  # Runs per task for reproducibility
TASK_IDS = [
    "phantom_volume_cleanup",
    "latency_spike_remediation",
    "noisy_neighbor_incident",
]

SYSTEM_PROMPT = """You are an expert Cloud SRE (Site Reliability Engineer) AI agent.
You manage a simulated cloud infrastructure. Your goal is to diagnose issues,
optimize costs, and restore services — all without causing collateral damage.

You will receive the current state of the infrastructure as JSON, including:
- resources: list of cloud resources (EC2, RDS, EBS, ALB) with their status, CPU, cost, tags
- alerts: active monitoring alerts with severity and messages
- total_hourly_cost: current spend rate
- system_uptime: overall health (0-100)

You must respond with a single JSON action object:
{
    "command": "terminate" | "scale" | "reboot" | "inspect" | "wait",
    "resource_id": "<id of the resource to act on, or null>",
    "params": {"target_size": "<new size>"} // only for scale command
}

RULES:
1. Always INSPECT a suspicious resource before terminating it
2. Never terminate production resources unless absolutely certain
3. Scale databases to fix performance issues
4. Reboot stopped instances that should be running
5. Wait only if no action is needed

Think step by step about the current state and alerts before choosing an action."""


def observation_to_prompt(obs, task_description: str, step: int) -> str:
    """Convert an Observation to a text prompt for the LLM."""
    obs_dict = obs.model_dump()

    # Simplify for readability
    resources_summary = []
    for r in obs_dict["resources"]:
        r_info = {
            "id": r["id"],
            "name": r["name"],
            "type": r["type"],
            "status": r["status"],
            "instance_size": r["instance_size"],
            "cpu%": r["cpu_utilization"],
            "cost_per_hour": f"${r['cost_per_hour']:.4f}",
            "tags": r["tags"],
        }
        if r["attached_to"]:
            r_info["attached_to"] = r["attached_to"]
        resources_summary.append(r_info)

    alerts_summary = []
    for a in obs_dict["alerts"]:
        alerts_summary.append({
            "severity": a["severity"],
            "message": a["message"],
            "resource_id": a.get("resource_id"),
        })

    prompt = f"""## Task
{task_description}

## Current Infrastructure State (Step {step}/{obs_dict['max_steps']})

**Total Hourly Cost:** ${obs_dict['total_hourly_cost']:.4f}/hr
**System Uptime:** {obs_dict['system_uptime']:.1f}%
{"**Budget Limit:** $" + f"{obs_dict['budget_limit']:.2f}/hr" if obs_dict.get('budget_limit') else ""}

### Resources
```json
{json.dumps(resources_summary, indent=2)}
```

### Active Alerts
```json
{json.dumps(alerts_summary, indent=2)}
```

Analyze the situation and respond with your next action as a JSON object."""

    return prompt


def parse_action(response_text: str) -> Action:
    """Parse the LLM response into an Action object."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find a JSON object in the text
        import re
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return Action(command=ActionCommand.WAIT)
        else:
            return Action(command=ActionCommand.WAIT)

    command = data.get("command", "wait")
    try:
        command_enum = ActionCommand(command)
    except ValueError:
        command_enum = ActionCommand.WAIT

    return Action(
        command=command_enum,
        resource_id=data.get("resource_id"),
        params=data.get("params", {}),
    )


def run_single_episode(client: OpenAI, env: CloudSREEnv, task_id: str) -> dict:
    """Run a single episode of a task. Returns results dict."""
    obs = env.reset(task_id)
    task_desc = env.get_task_description()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    step = 0
    done = False
    total_step_reward = 0.0

    while not done:
        step += 1
        user_prompt = observation_to_prompt(obs, task_desc, step)
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            assistant_msg = response.choices[0].message.content
        except Exception as e:
            print(f"  [API Error at step {step}]: {e}")
            assistant_msg = '{"command": "wait"}'

        messages.append({"role": "assistant", "content": assistant_msg})

        action = parse_action(assistant_msg)
        result = env.step(action)

        obs = result.observation
        total_step_reward += result.reward
        done = result.done

        action_str = f"{action.command.value}({action.resource_id or ''})"
        print(f"  Step {step}: {action_str} → reward={result.reward:+.4f} | "
              f"cost=${obs.total_hourly_cost:.4f}/hr | uptime={obs.system_uptime:.1f}%")

        if done:
            final_score = result.info.get("final_score", 0.0)
            breakdown = result.info.get("grading_breakdown", {})
            return {
                "task_id": task_id,
                "steps": step,
                "final_score": final_score,
                "cumulative_step_reward": round(total_step_reward, 4),
                "grading_breakdown": breakdown,
            }

    # Should not reach here, but just in case
    score, breakdown = env.grade()
    return {
        "task_id": task_id,
        "steps": step,
        "final_score": score,
        "cumulative_step_reward": round(total_step_reward, 4),
        "grading_breakdown": breakdown,
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Usage: export OPENAI_API_KEY='sk-...' && python baseline.py")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    env = CloudSREEnv(max_steps=15)

    print("=" * 70)
    print("  Cloud SRE OpenEnv -- Baseline Agent (GPT-4o)")
    print("=" * 70)
    print()

    all_results = {}

    for task_id in TASK_IDS:
        task_scores = []
        print(f"--- Task: {task_id} ---")

        for run in range(1, NUM_RUNS + 1):
            print(f"\n  Run {run}/{NUM_RUNS}:")
            result = run_single_episode(client, env, task_id)
            task_scores.append(result["final_score"])
            print(f"  → Final Score: {result['final_score']:.2f}")
            print(f"  → Breakdown: {result['grading_breakdown']}")

        avg_score = sum(task_scores) / len(task_scores)
        all_results[task_id] = {
            "individual_scores": task_scores,
            "average_score": round(avg_score, 4),
        }
        print(f"\n  Average Score: {avg_score:.4f}")
        print()

    # ── Final Report ──
    print("=" * 70)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Task':<35} {'Avg Score':>10} {'Runs':>8}")
    print("-" * 55)

    overall_scores = []
    for task_id, data in all_results.items():
        avg = data["average_score"]
        overall_scores.append(avg)
        print(f"{task_id:<35} {avg:>10.4f} {NUM_RUNS:>8}")

    overall_avg = sum(overall_scores) / len(overall_scores)
    print("-" * 55)
    print(f"{'OVERALL AVERAGE':<35} {overall_avg:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
