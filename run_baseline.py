"""Quick single-run baseline to capture scores."""
import os, json, sys
from openai import OpenAI
from env import CloudSREEnv  # type: ignore
from models import Action, ActionCommand  # type: ignore

api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
if not api_key:
    print("ERROR: set API_KEY or HF_TOKEN first"); sys.exit(1)

client = OpenAI(base_url=api_base_url, api_key=api_key)
env = CloudSREEnv(max_steps=15)
tasks = ["phantom_volume_cleanup", "latency_spike_remediation", "noisy_neighbor_incident"]

SYSTEM = (
    "You are a Cloud SRE agent. Reply ONLY with a JSON action: "
    '{"command":"<terminate/scale/reboot/inspect/wait>","resource_id":"<id or null>","params":{"target_size":"<size>"}}. '
    "INSPECT suspicious resources first. Never kill production instances. "
    "Scale databases to fix latency. Reboot stopped production instances."
)

results = {}
for task_id in tasks:
    obs = env.reset(task_id)
    desc = env.get_task_description()
    done = False
    step = 0
    messages = [{"role": "system", "content": SYSTEM}]

    while not done:
        step += 1
        state = obs.model_dump()
        resources = [
            {"id": r["id"], "type": r["type"], "status": r["status"],
             "size": r["instance_size"], "cpu": r["cpu_utilization"],
             "cost": r["cost_per_hour"], "tags": r["tags"],
             "attached_to": r.get("attached_to")}
            for r in state["resources"]
        ]
        alerts = [
            {"severity": a["severity"], "message": a["message"],
             "resource_id": a.get("resource_id")}
            for a in state["alerts"]
        ]
        prompt = (
            f"Task: {desc}\n"
            f"Resources: {json.dumps(resources)}\n"
            f"Alerts: {json.dumps(alerts)}\n"
            f"Cost: ${state['total_hourly_cost']:.4f}/hr\n"
            f"Uptime: {state['system_uptime']:.1f}%\n"
            f"Step {step}/{state['max_steps']}"
        )
        messages.append({"role": "user", "content": prompt})

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            data = json.loads(content)
        except Exception as e:
            print(f"  API Error step {step}: {e}")
            data = {"command": "wait"}
            messages.append({"role": "assistant", "content": '{"command":"wait"}'})

        cmd = ActionCommand(data.get("command", "wait"))
        action = Action(
            command=cmd,
            resource_id=data.get("resource_id"),
            params=data.get("params", {}),
        )
        result = env.step(action)
        obs = result.observation
        done = result.done

        rid = data.get("resource_id", "")
        print(f"  [{task_id}] step {step}: {cmd.value}({rid}) -> reward={result.reward:+.4f}")

        if done:
            score = result.info.get("final_score", 0)
            breakdown = result.info.get("grading_breakdown", {})
            results[task_id] = {"score": score, "steps": step, "breakdown": breakdown}
            print(f"  SCORE: {score:.2f}")
            print(f"  BREAKDOWN: {json.dumps(breakdown)}")

print("\n" + "=" * 60)
print("BASELINE RESULTS (GPT-4o, temperature=0.0)")
print("=" * 60)
total = 0
for tid, r in results.items():
    print(f"  {tid}: {r['score']:.2f} ({r['steps']} steps)")
    total += r["score"]
avg = total / len(results) if results else 0
print(f"  OVERALL AVERAGE: {avg:.2f}")
print("=" * 60)
