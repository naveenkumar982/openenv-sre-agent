"""Quick single-run baseline to capture scores using urllib to bypass httpx issues."""
import os, json, sys, urllib.request, urllib.error
from env import CloudSREEnv
from models import Action, ActionCommand

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # Use the hardcoded key as fallback for this specific run if env var is missing
    api_key = "YOUR_API_KEY_HERE"

env = CloudSREEnv(max_steps=15)
tasks = ["phantom_volume_cleanup", "latency_spike_remediation", "noisy_neighbor_incident"]

SYSTEM = (
    "You are a Cloud SRE agent. Reply ONLY with a JSON action: "
    '{"command":"<terminate/scale/reboot/inspect/wait>","resource_id":"<id or null>","params":{"target_size":"<size>"}}. '
    "INSPECT suspicious resources first. Never kill production instances. "
    "Scale databases to fix latency. Reboot stopped production instances."
)

def chat_complete(messages):
    url = "https://api.openai.com/v1/chat/completions"
    data = json.dumps({
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"}
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  API Error (HTTP {e.code}): {body}")
        return '{"command": "wait"}'
    except Exception as e:
        print(f"  API Error: {e}")
        return '{"command": "wait"}'

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

        content = chat_complete(messages)
        messages.append({"role": "assistant", "content": content})
        
        try:
            data = json.loads(content)
        except:
            data = {"command": "wait"}

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
