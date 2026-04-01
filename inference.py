"""
Baseline inference agent for the Cloud SRE OpenEnv.

Demonstrates the agent loop: reset() -> step() -> ... -> done
using simple heuristic rules for each task.

Usage:
    # Against a running server:
    python inference.py --url http://localhost:7860

    # Direct (no server needed):
    python inference.py --direct
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Direct mode (no HTTP, import env directly) ──────────────────────────────

def run_direct():
    """Run inference directly using the Python environment."""
    from env import CloudSREEnv
    from models import Action, ActionCommand
    from tasks import list_tasks

    env = CloudSREEnv(max_steps=15)
    all_tasks = list_tasks()

    print("=" * 60)
    print("  Cloud SRE OpenEnv — Baseline Inference Agent (Direct)")
    print("=" * 60)

    for task_info in all_tasks:
        task_id = task_info["id"]
        print(f"\n{'-' * 50}")
        print(f"  Task: {task_id} ({task_info['difficulty']})")
        print(f"{'-' * 50}")

        obs = env.reset(task_id)
        print(f"  Initial: {len(obs.resources)} resources, "
              f"${obs.total_hourly_cost:.4f}/hr, "
              f"uptime={obs.system_uptime:.1f}%")

        actions = get_heuristic_actions(task_id, obs)

        for i, action in enumerate(actions):
            result = env.step(action)
            cmd_str = f"{action.command.value}({action.resource_id or ''})"
            print(f"  Step {i+1}: {cmd_str} -> reward={result.reward:+.4f}")

            if result.done:
                score = result.info.get("final_score", 0)
                print(f"  >> Episode done. Final score: {score:.2f}/1.00")
                break

        # Run to completion if not already done
        if not result.done:
            while not result.done:
                result = env.step(Action(command=ActionCommand.WAIT))

            score = result.info.get("final_score", 0)
            print(f"  >> Episode done (waited). Final score: {score:.2f}/1.00")

        score, breakdown = env.grade()
        print(f"  Grading: {json.dumps(breakdown, indent=4)}")

    print(f"\n{'=' * 60}")
    print("  Baseline inference complete.")
    print(f"{'=' * 60}")


# ─── HTTP mode (call the server endpoints) ───────────────────────────────────

def run_http(base_url: str):
    """Run inference against a running OpenEnv HTTP server."""
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        print("ERROR: urllib not available")
        sys.exit(1)

    print("=" * 60)
    print("  Cloud SRE OpenEnv — Baseline Inference Agent (HTTP)")
    print(f"  Server: {base_url}")
    print("=" * 60)

    # Health check
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req) as resp:
            health = json.loads(resp.read().decode())
            print(f"  Health: {health['status']}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server: {e}")
        sys.exit(1)

    # Get tasks
    try:
        req = urllib.request.Request(f"{base_url}/tasks")
        with urllib.request.urlopen(req) as resp:
            tasks = json.loads(resp.read().decode())
    except Exception:
        tasks = [
            {"id": "phantom_volume_cleanup"},
            {"id": "latency_spike_remediation"},
            {"id": "noisy_neighbor_incident"},
        ]

    for task_info in tasks:
        task_id = task_info["id"]
        print(f"\n{'-' * 50}")
        print(f"  Task: {task_id}")
        print(f"{'-' * 50}")

        # Reset
        reset_data = json.dumps({"task_id": task_id}).encode()
        req = urllib.request.Request(
            f"{base_url}/reset",
            data=reset_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            reset_result = json.loads(resp.read().decode())
            obs = reset_result["observation"]
            print(f"  Reset OK: {len(obs.get('resources', []))} resources")

        # Get heuristic actions for this task
        heuristic_actions = get_heuristic_actions_dict(task_id)

        for i, action_data in enumerate(heuristic_actions):
            step_data = json.dumps({"action": action_data}).encode()
            req = urllib.request.Request(
                f"{base_url}/step",
                data=step_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                step_result = json.loads(resp.read().decode())
                reward = step_result.get("reward", 0)
                done = step_result.get("done", False)
                cmd_str = f"{action_data['command']}({action_data.get('resource_id', '')})"
                print(f"  Step {i+1}: {cmd_str} -> reward={reward:+.4f}")

                if done:
                    score = step_result.get("info", {}).get("final_score", 0)
                    print(f"  >> Episode done. Final score: {score:.2f}/1.00")
                    break

        # Wait until done
        if not done:
            while not done:
                wait_data = json.dumps({"action": {"command": "wait"}}).encode()
                req = urllib.request.Request(
                    f"{base_url}/step",
                    data=wait_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req) as resp:
                    step_result = json.loads(resp.read().decode())
                    done = step_result.get("done", False)

            score = step_result.get("info", {}).get("final_score", 0)
            print(f"  >> Episode done (waited). Final score: {score:.2f}/1.00")

    print(f"\n{'=' * 60}")
    print("  Baseline inference complete.")
    print(f"{'=' * 60}")


# ─── Heuristic Action Selection ──────────────────────────────────────────────

def get_heuristic_actions(task_id, obs):
    """Return a list of Action objects based on simple heuristics."""
    from models import Action, ActionCommand

    if task_id == "phantom_volume_cleanup":
        # Find and terminate unattached EBS volumes
        actions = []
        for r in obs.resources:
            if r.type.value == "ebs_volume" and r.status.value == "available":
                actions.append(Action(command=ActionCommand.TERMINATE, resource_id=r.id))
        return actions

    elif task_id == "latency_spike_remediation":
        # Find the under-provisioned RDS and scale it up
        actions = []
        for r in obs.resources:
            if r.type.value == "rds_database" and r.cpu_utilization > 90:
                actions.append(Action(
                    command=ActionCommand.SCALE,
                    resource_id=r.id,
                    params={"target_size": "db.t3.medium"},
                ))
        return actions

    elif task_id == "noisy_neighbor_incident":
        # Inspect rogue, terminate it, reboot crashed prod
        actions = []
        rogue_id = None
        crashed_id = None
        for r in obs.resources:
            if r.tags.get("env") == "test" and r.cpu_utilization >= 95:
                rogue_id = r.id
            if r.status.value == "stopped" and r.tags.get("env") == "prod":
                crashed_id = r.id

        if rogue_id:
            actions.append(Action(command=ActionCommand.INSPECT, resource_id=rogue_id))
            actions.append(Action(command=ActionCommand.TERMINATE, resource_id=rogue_id))
        if crashed_id:
            actions.append(Action(command=ActionCommand.REBOOT, resource_id=crashed_id))
        return actions

    return []


def get_heuristic_actions_dict(task_id):
    """Return a list of action dicts for HTTP mode."""
    if task_id == "phantom_volume_cleanup":
        return [
            {"command": "terminate", "resource_id": "ebs-orphan-001"},
            {"command": "terminate", "resource_id": "ebs-orphan-002"},
            {"command": "terminate", "resource_id": "ebs-orphan-003"},
        ]
    elif task_id == "latency_spike_remediation":
        return [
            {"command": "scale", "resource_id": "rds-primary-001",
             "params": {"target_size": "db.t3.medium"}},
        ]
    elif task_id == "noisy_neighbor_incident":
        return [
            {"command": "inspect", "resource_id": "ec2-rogue-test-001"},
            {"command": "terminate", "resource_id": "ec2-rogue-test-001"},
            {"command": "reboot", "resource_id": "ec2-backend-prod-001"},
        ]
    return []


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud SRE OpenEnv Baseline Agent")
    parser.add_argument("--url", type=str, default=None,
                        help="Base URL of the running server (e.g. http://localhost:7860)")
    parser.add_argument("--direct", action="store_true",
                        help="Run directly without HTTP server")
    args = parser.parse_args()

    if args.direct or args.url is None:
        run_direct()
    else:
        run_http(args.url)
