"""
ReAct Agent for the Cloud SRE OpenEnv.

Implements the Think -> Act -> Observe reasoning loop with full trace
visibility. Uses OpenAI GPT models with structured reasoning output.

Usage:
    from react_agent import ReActAgent
    agent = ReActAgent(api_key="sk-...")
    trace = agent.run_episode(env, task_id="noisy_neighbor_incident")
    print(trace.to_markdown())
"""

import json
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


# ─── ReAct Trace Data Structures ─────────────────────────────────────────────

@dataclass
class ReActStep:
    """A single Think -> Act -> Observe step."""
    step_number: int
    thought: str
    action_text: str
    action_json: Dict[str, Any]
    observation: str
    reward: float
    cumulative_reward: float
    timestamp: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"### Step {self.step_number}",
            f"**🧠 Thought:** {self.thought}",
            "",
            f"**⚡ Action:** `{self.action_text}`",
            "",
            f"**👁 Observation:** {self.observation[:300]}{'...' if len(self.observation) > 300 else ''}",
            "",
            f"**📊 Reward:** `{self.reward:+.4f}` (cumulative: `{self.cumulative_reward:+.4f}`)",
            "",
            "---",
        ]
        return "\n".join(lines)


@dataclass
class ReActTrace:
    """Full reasoning trace for an episode."""
    task_id: str
    model: str
    steps: List[ReActStep] = field(default_factory=list)
    final_score: float = 0.0
    grading_breakdown: Dict[str, Any] = field(default_factory=dict)
    initial_cost: float = 0.0
    final_cost: float = 0.0
    total_time: float = 0.0

    def add_step(self, step: ReActStep):
        self.steps.append(step)

    def to_markdown(self) -> str:
        header = [
            f"## ReAct Agent Trace — {self.task_id}",
            f"**Model:** {self.model} | **Steps:** {len(self.steps)} | **Score:** {self.final_score:.2f}",
            f"**Cost:** ${self.initial_cost:.4f}/hr → ${self.final_cost:.4f}/hr | **Time:** {self.total_time:.1f}s",
            "",
            "---",
            "",
        ]
        step_texts = [s.to_markdown() for s in self.steps]
        return "\n".join(header + step_texts)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "steps_used": len(self.steps),
            "final_score": self.final_score,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "cost_saved": round(self.initial_cost - self.final_cost, 4),
            "grading_breakdown": self.grading_breakdown,
        }


# ─── System Prompts ──────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an expert Cloud SRE (Site Reliability Engineer) AI Agent using the ReAct framework.

For each step you MUST respond in EXACTLY this format:

**Thought**: [Your detailed reasoning about the current situation — what you observe, what needs attention, what risks exist, and what action makes sense]

**Action**: [A single JSON object with the action to take]

Available actions:
- {"command": "inspect", "resource_id": "<id>"} — Get detailed info about a resource (always do this first for suspicious resources)
- {"command": "terminate", "resource_id": "<id>"} — Remove a resource permanently
- {"command": "scale", "resource_id": "<id>", "params": {"target_size": "<size>"}} — Resize (valid RDS sizes: db.t3.micro, db.t3.small, db.t3.medium, db.t3.large, db.t3.xlarge)
- {"command": "reboot", "resource_id": "<id>"} — Restart a stopped instance
- {"command": "wait"} — Take no action this step
- {"command": "analyze_costs"} — Get a detailed cost breakdown (free action, doesn't use a step)
- {"command": "check_alerts"} — Get a summary of all alerts (free action, doesn't use a step)

CRITICAL RULES:
1. ALWAYS inspect a resource before terminating it — gather evidence first
2. NEVER terminate production (env:prod) resources unless absolutely certain they are the root cause
3. Look at tags carefully — env:test vs env:prod matters enormously
4. Think about cascading effects: will terminating this resource affect other services?
5. For latency issues, scaling the database is usually the right fix
6. For cost optimization, target unattached/idle resources
7. For incidents, investigate first (inspect), then act decisively
8. Minimize the number of actions — efficiency matters

Think step by step. Be precise."""


# ─── ReAct Agent ──────────────────────────────────────────────────────────────

class ReActAgent:
    """
    ReAct-style agent that runs a Think -> Act -> Observe loop
    against the CloudSREEnv, with full reasoning trace capture.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
        self.base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model = model or os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def run_episode(self, env, task_id: str, seed: Optional[int] = None,
                    on_step=None) -> ReActTrace:
        """
        Run a full episode with ReAct reasoning.

        Args:
            env: CloudSREEnv instance
            task_id: Task to run
            seed: Optional seed for procedural generation
            on_step: Optional callback(trace_so_far) called after each step

        Returns:
            Complete ReActTrace
        """
        start_time = time.time()
        obs = env.reset(task_id, seed=seed)
        task_desc = env.get_task_description()

        trace = ReActTrace(
            task_id=task_id,
            model=self.model,
            initial_cost=obs.total_hourly_cost,
        )

        messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        done = False
        cumulative_reward = 0.0

        while not done:
            # Build user message with current state
            user_msg = self._format_observation(obs, task_desc, len(trace.steps) + 1)
            messages.append({"role": "user", "content": user_msg})

            # Call LLM
            thought, action_json = self._get_react_response(messages)

            # Check for virtual tools (analyze_costs, check_alerts)
            cmd = action_json.get("command", "wait")
            if cmd in ("analyze_costs", "check_alerts"):
                # Virtual tool — get result, add to conversation, loop again
                if cmd == "analyze_costs":
                    tool_result = env.analyze_costs()
                elif cmd == "check_alerts":
                    tool_result = env.check_alerts()

                virtual_step = ReActStep(
                    step_number=len(trace.steps) + 1,
                    thought=thought,
                    action_text=cmd,
                    action_json=action_json,
                    observation=f"[Tool Result]\n{tool_result}",
                    reward=0.0,
                    cumulative_reward=cumulative_reward,
                    timestamp=time.time(),
                )
                trace.add_step(virtual_step)
                if on_step:
                    on_step(trace)

                # Add to conversation and continue
                messages.append({"role": "assistant", "content": f"**Thought**: {thought}\n\n**Action**: {json.dumps(action_json)}"})
                messages.append({"role": "user", "content": f"**Observation**: {tool_result}\n\nContinue with the next action."})
                continue

            # Execute real action in environment
            from models import Action, ActionCommand
            try:
                command_enum = ActionCommand(cmd)
            except ValueError:
                command_enum = ActionCommand.WAIT

            action = Action(
                command=command_enum,
                resource_id=action_json.get("resource_id"),
                params=action_json.get("params", {}),
            )

            result = env.step(action)
            obs = result.observation
            cumulative_reward += result.reward
            done = result.done

            # Format observation text
            action_text = f"{cmd}({action_json.get('resource_id', '')})"
            obs_text = result.info.get("action_result", "Action completed.")
            if result.info.get("chaos_event"):
                obs_text += f"\n{result.info['chaos_event']}"

            step = ReActStep(
                step_number=len(trace.steps) + 1,
                thought=thought,
                action_text=action_text,
                action_json=action_json,
                observation=obs_text,
                reward=result.reward,
                cumulative_reward=cumulative_reward,
                timestamp=time.time(),
            )
            trace.add_step(step)

            if on_step:
                on_step(trace)

            # Add to conversation
            messages.append({"role": "assistant", "content": f"**Thought**: {thought}\n\n**Action**: {json.dumps(action_json)}"})
            messages.append({"role": "user", "content": f"**Observation**: {obs_text}"})

            # Get final results
            if done:
                trace.final_score = result.info.get("final_score", 0.0)
                trace.grading_breakdown = result.info.get("grading_breakdown", {})
                trace.final_cost = obs.total_hourly_cost

        trace.total_time = time.time() - start_time
        return trace

    def _format_observation(self, obs, task_description: str, step: int) -> str:
        """Format observation into a prompt for the LLM."""
        obs_dict = obs.model_dump()

        resources_summary = []
        for r in obs_dict["resources"]:
            r_info = {
                "id": r["id"], "name": r["name"], "type": r["type"],
                "status": r["status"], "instance_size": r["instance_size"],
                "cpu%": r["cpu_utilization"], "memory%": r["memory_utilization"],
                "cost_per_hour": f"${r['cost_per_hour']:.4f}",
                "tags": r["tags"],
            }
            if r.get("attached_to"):
                r_info["attached_to"] = r["attached_to"]
            resources_summary.append(r_info)

        alerts_summary = []
        for a in obs_dict["alerts"]:
            alerts_summary.append({
                "severity": a["severity"], "message": a["message"],
                "resource_id": a.get("resource_id"),
            })

        budget_line = ""
        if obs_dict.get("budget_limit"):
            budget_line = f"\n**Budget Limit:** ${obs_dict['budget_limit']:.2f}/hr"

        prompt = f"""## Task
{task_description}

## Current Infrastructure State (Step {step}/{obs_dict['max_steps']})

**Total Hourly Cost:** ${obs_dict['total_hourly_cost']:.4f}/hr
**System Uptime:** {obs_dict['system_uptime']:.1f}%{budget_line}

### Resources ({len(resources_summary)} total)
```json
{json.dumps(resources_summary, indent=2)}
```

### Active Alerts ({len(alerts_summary)})
```json
{json.dumps(alerts_summary, indent=2)}
```

Analyze the situation step by step. Respond with your **Thought** and **Action**."""
        return prompt

    def _get_react_response(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Get a ReAct-formatted response from the LLM."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
            )
            content = response.choices[0].message.content or ""
            return self._parse_react_response(content)
        except Exception as e:
            return f"Error calling LLM: {e}", {"command": "wait"}

    def _parse_react_response(self, text: str) -> Tuple[str, Dict]:
        """Parse Thought and Action from the ReAct response."""
        thought = ""
        action_json = {"command": "wait"}

        # Extract Thought
        thought_match = re.search(
            r'\*\*Thought\*\*:?\s*(.*?)(?=\*\*Action\*\*|$)',
            text, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action JSON
        action_match = re.search(
            r'\*\*Action\*\*:?\s*(.*)',
            text, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action_text = action_match.group(1).strip()
            # Find JSON in the action text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', action_text)
            if json_match:
                try:
                    action_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try simpler JSON
                    simple_match = re.search(r'\{[^{}]*\}', action_text)
                    if simple_match:
                        try:
                            action_json = json.loads(simple_match.group())
                        except json.JSONDecodeError:
                            pass

        return thought, action_json
