import os
import json
import sys
from typing import List, Optional
from openai import OpenAI

from env import CloudSREEnv
from models import Action, ActionCommand
from tasks import list_tasks

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "cloud-sre-simulator"
SUCCESS_SCORE_THRESHOLD = 0.5  # Treat >= 0.5 as success

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = (
    "You are an expert Cloud SRE agent resolving infrastructure incidents.\n"
    "Reply ONLY with a raw JSON action, no markdown formatting, no code blocks.\n"
    'Format: {"command":"<terminate/scale/reboot/inspect/wait>","resource_id":"<id or null>","params":{"target_size":"<size>"}}\n'
    "\n"
    "CRITICAL RULES:\n"
    "1. For id 'phantom_volume_cleanup', find and terminate unattached/available EBS volumes.\n"
    "2. For id 'latency_spike_remediation', scale up under-provisioned RDS reading high CPU.\n"
    "3. For id 'noisy_neighbor_incident', terminate rogue test EC2 and reboot crashed prod EC2.\n"
    "4. NEVER terminate production resources unless explicitly rogue."
)

def run_tests():
    if not API_KEY:
        print("ERROR: API_KEY or HF_TOKEN is not set", file=sys.stderr)
        # Try to use a dummy key if none provided and running locally, but evaluator will inject it.
        pass

    # Ensure client picks up proper BASE URL
    # Initialize per instructions: base_url=os.environ["API_BASE_URL"] and api_key=os.environ["API_KEY"]
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    
    env = CloudSREEnv(max_steps=15)
    all_tasks = list_tasks()

    for task_info in all_tasks:
        task_id = task_info["id"]
        
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        obs = env.reset(task_id)
        desc = env.get_task_description()
        
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False
        done = False
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        while not done:
            steps_taken += 1
            state = obs.model_dump()
            
            resources = []
            for r in state["resources"]:
                r_dict = {
                    "id": r["id"], 
                    "type": r["type"], 
                    "status": r["status"],
                    "size": r["instance_size"], 
                    "cpu": r["cpu_utilization"],
                    "tags": r["tags"]
                }
                if r.get("attached_to"):
                    r_dict["attached_to"] = r.get("attached_to")
                resources.append(r_dict)
                
            prompt = (
                f"Task ID: {task_id}\n"
                f"Description: {desc}\n"
                f"Resources: {json.dumps(resources)}\n"
                f"Step {steps_taken}/{state['max_steps']}\n"
                "What is your next action JSON?"
            )
            messages.append({"role": "user", "content": prompt})

            action_str = ""
            error_msg = None
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=200,
                    # DO NOT use response_format={"type": "json_object"} as generic HF routers may not support it universally
                )
                content = (resp.choices[0].message.content or "").strip()
                # Clean up any potential markdown formatting the model might mistakenly include
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                action_str = content
                messages.append({"role": "assistant", "content": content})
                data = json.loads(content)
            except Exception as e:
                error_msg = f"LLM Error: {str(e)}"
                data = {"command": "wait"}
                action_str = json.dumps(data)
                messages.append({"role": "assistant", "content": action_str})

            try:
                cmd = ActionCommand(data.get("command", "wait"))
                action = Action(
                    command=cmd,
                    resource_id=data.get("resource_id"),
                    params=data.get("params", {}),
                )
            except Exception as e:
                action = Action(command=ActionCommand.WAIT)
                error_msg = f"Parse Error: {str(e)}"
                
            result = env.step(action)
            obs = result.observation
            done = result.done
            reward = result.reward

            rewards.append(reward)
            
            # format action string safely (no newlines)
            safe_action_str = action_str.replace('\n', ' ').replace('\r', '')
            log_step(step=steps_taken, action=safe_action_str, reward=reward, done=done, error=error_msg)

            if done:
                score = result.info.get("final_score", 0.0)
                score = min(max(score, 0.0), 1.0) # Clamp to [0, 1]
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
                
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    run_tests()
