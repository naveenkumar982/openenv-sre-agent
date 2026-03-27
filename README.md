---
title: Cloud SRE and FinOps Simulator
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: OpenEnv AI Agent Environment for Cloud SRE operations
---

# ☁️ Cloud SRE & FinOps Simulator — OpenEnv

An AI agent environment where the agent acts as a **Site Reliability Engineer (SRE)** managing simulated cloud infrastructure. The agent must diagnose outages, terminate idle resources, scale services, and optimize costs — all without causing collateral damage to production workloads.

Built on the **OpenEnv spec** for the [OpenEnv AI Hackathon](https://pytorch.org/) by Scaler × Meta × Hugging Face × PyTorch.

---

## 🌟 Why This Environment?

| Criteria | How We Score |
|----------|-------------|
| **Real-world utility** | Every tech company runs cloud infra ops. SRE/FinOps is a $100B+ industry. |
| **Novelty** | First OpenEnv environment modeling cloud infrastructure management. |
| **Reward design** | Dense per-step rewards (cost savings, penalties for collateral damage). |
| **Task quality** | 3 progressively harder tasks with deterministic, partial-credit graders. |
| **LLM-friendly** | JSON state → JSON action — perfect for GPT-4o / Llama structured reasoning. |

---

## 🎮 Tasks

### Task 1 — Easy: Phantom Volume Cleanup
> Identify and terminate 3 unattached, idle EBS volumes wasting $4.20/hr.

- **Score**: +0.33 per orphan removed, -0.50 per active resource destroyed
- **Max**: 1.00

### Task 2 — Medium: Latency Spike Remediation
> Scale up an under-provisioned RDS database to fix API latency, within a $12/hr budget.

- **Score**: +0.40 scaling, +0.30 under budget, +0.30 alert resolved, -0.30 per EC2 killed
- **Max**: 1.00

### Task 3 — Hard: Noisy Neighbor Incident
> A rogue test instance crashed the prod backend. Investigate, terminate the offender, restore the backend.

- **Score**: +0.20 inspect first, +0.30 terminate rogue, +0.30 reboot backend, +0.20 alerts resolved, -0.50 per prod instance killed
- **Max**: 1.00

---

## 🔧 Action & Observation Spaces

### Observation (JSON)
```json
{
    "resources": [
        {
            "id": "ec2-web-001",
            "type": "ec2_instance",
            "status": "running",
            "instance_size": "t3.medium",
            "cpu_utilization": 45.0,
            "cost_per_hour": 0.0416,
            "tags": {"env": "prod", "role": "web"}
        }
    ],
    "alerts": [
        {
            "severity": "critical",
            "message": "API p99 latency > 2000ms",
            "resource_id": "rds-primary-001"
        }
    ],
    "total_hourly_cost": 5.23,
    "system_uptime": 78.0
}
```

### Action (JSON)
```json
{
    "command": "terminate | scale | reboot | inspect | wait",
    "resource_id": "ec2-web-001",
    "params": {"target_size": "db.t3.medium"}
}
```

### Reward Design
- **Dense signals** every step (not just end-of-episode)
- **Positive**: terminating idle resources, scaling to fix alerts, rebooting crashed services
- **Negative**: killing prod instances, invalid actions, excessive waiting
- **Grader**: Deterministic end-of-episode score (0.0 – 1.0) based on specific criteria

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio UI
```bash
python app.py
# Open http://localhost:7860
```

### 3. Run the Baseline Agent
```bash
export OPENAI_API_KEY="sk-..."
python baseline.py
```

### 4. Docker
```bash
docker build -t cloud-sre-env .
docker run -p 7860:7860 cloud-sre-env
```

---

## 📊 Baseline Scores (GPT-4o)

| Task | Difficulty | Avg Score |
|------|-----------|-----------|
| Phantom Volume Cleanup | Easy | ~0.80 |
| Latency Spike Remediation | Medium | ~0.70 |
| Noisy Neighbor Incident | Hard | ~0.50 |
| **Overall** | | **~0.67** |

*(Scores averaged over 3 runs with temperature=0.0)*

---

## 📁 Project Structure

```
openenv-sre-agent/
├── models.py           # Pydantic typed models (Observation, Action, Resource, Alert)
├── env.py              # Core OpenEnv environment (step/reset/state)
├── tasks.py            # 3 task definitions + deterministic graders
├── baseline.py         # GPT-4o baseline agent script
├── app.py              # Gradio UI for Hugging Face Spaces
├── openenv.yaml        # OpenEnv spec metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container for HF Spaces deployment
└── README.md           # This file
```

---

## 🔬 OpenEnv Spec Compliance

- ✅ `step(action)` → returns `(observation, reward, done, info)`
- ✅ `reset(task_id)` → returns initial observation
- ✅ `state()` → returns current observation
- ✅ Typed Pydantic models for all data
- ✅ `openenv.yaml` with full metadata
- ✅ 3 tasks with difficulty range (easy → medium → hard)
- ✅ Deterministic graders (0.0 – 1.0)
- ✅ Dense reward function with partial progress signals
- ✅ Baseline script with reproducible scores
- ✅ Dockerfile + HF Spaces deployment

---

## 📜 License

MIT License — built for the OpenEnv AI Hackathon 2025.
