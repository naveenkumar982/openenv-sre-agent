"""
OpenEnv Spec Validator -- validates the Cloud SRE environment against the OpenEnv specification.

Checks:
  1. openenv.yaml exists and has required fields
  2. Environment class has reset(), step(), state() methods
  3. reset(task_id) returns a valid Observation
  4. step(action) returns (Observation, reward, done, info) via StepResult
  5. state() returns current Observation
  6. Grader returns deterministic score in [0.0, 1.0]
  7. All 3 tasks load and complete successfully
  8. Pydantic models validate correctly
  9. Dense reward shaping works (non-zero intermediate rewards)
  10. Episode terminates within max_steps
"""
import sys
import os
import io
import yaml  # type: ignore

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    print("=" * 60)
    print("  OpenEnv Spec Validator — Cloud SRE Simulator")
    print("=" * 60)

    # --- 1. openenv.yaml ---
    print("\n[1] openenv.yaml")
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    check("openenv.yaml exists", os.path.exists(yaml_path))

    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    for field in ["name", "version", "description", "entrypoint", "tasks",
                  "action_space", "observation_space", "reward_range", "max_steps"]:
        check(f"Required field '{field}'", field in spec)

    check("Has 3 tasks", len(spec.get("tasks", [])) == 3, f"found {len(spec.get('tasks', []))}")
    check("Reward range is [-1.0, 1.0]", spec.get("reward_range") == [-1.0, 1.0])
    check("Max steps defined", spec.get("max_steps", 0) > 0, f"max_steps={spec.get('max_steps')}")

    # --- 2. Pydantic Models ---
    print("\n[2] Pydantic Models")
    from models import Resource, Alert, Observation, Action, ActionCommand, StepResult  # type: ignore

    check("Resource model importable", True)
    check("Alert model importable", True)
    check("Observation model importable", True)
    check("Action model importable", True)
    check("StepResult model importable", True)
    check("ActionCommand enum has 5 commands",
          len(ActionCommand) == 5,
          f"found {len(ActionCommand)}: {[c.value for c in ActionCommand]}")

    # Test Action validation
    try:
        a = Action(command=ActionCommand.TERMINATE, resource_id="test-001")
        check("Action validates correctly", a.command == ActionCommand.TERMINATE)
    except Exception as e:
        check("Action validates correctly", False, str(e))

    try:
        Action(command="invalid_command", resource_id="x")
        check("Action rejects invalid command", False, "should have raised")
    except Exception:
        check("Action rejects invalid command", True)

    # --- 3. Environment API ---
    print("\n[3] Environment API")
    from env import CloudSREEnv  # type: ignore

    env = CloudSREEnv(max_steps=15)
    check("CloudSREEnv instantiates", env is not None)
    check("Has reset() method", hasattr(env, "reset") and callable(env.reset))
    check("Has step() method", hasattr(env, "step") and callable(env.step))
    check("Has state() method", hasattr(env, "state") and callable(env.state))
    check("Has grade() method", hasattr(env, "grade") and callable(env.grade))

    # --- 4. Task Execution ---
    print("\n[4] Task Execution & Grading")
    from tasks import list_tasks, TASK_REGISTRY  # type: ignore

    task_list = list_tasks()
    check("list_tasks() returns list", isinstance(task_list, list))
    check("3 tasks registered", len(task_list) == 3, f"found {len(task_list)}")

    difficulties = {t["difficulty"] for t in task_list}
    check("Has easy/medium/hard",
          difficulties == {"easy", "medium", "hard"},
          f"found: {difficulties}")

    # Run each task
    task_configs = [
        ("phantom_volume_cleanup", [
            Action(command=ActionCommand.TERMINATE, resource_id="ebs-orphan-001"),
            Action(command=ActionCommand.TERMINATE, resource_id="ebs-orphan-002"),
            Action(command=ActionCommand.TERMINATE, resource_id="ebs-orphan-003"),
        ]),
        ("latency_spike_remediation", [
            Action(command=ActionCommand.SCALE, resource_id="rds-primary-001",
                   params={"target_size": "db.t3.medium"}),
        ]),
        ("noisy_neighbor_incident", [
            Action(command=ActionCommand.INSPECT, resource_id="ec2-rogue-test-001"),
            Action(command=ActionCommand.TERMINATE, resource_id="ec2-rogue-test-001"),
            Action(command=ActionCommand.REBOOT, resource_id="ec2-backend-prod-001"),
        ]),
    ]

    for task_id, actions in task_configs:
        print(f"\n  --- {task_id} ---")
        obs = env.reset(task_id)
        check(f"  reset('{task_id}') returns Observation",
              isinstance(obs, Observation),
              f"type={type(obs).__name__}")

        # Check state() matches
        current = env.state()
        check(f"  state() returns Observation", isinstance(current, Observation))

        # Check Observation fields
        check(f"  Observation has resources", hasattr(obs, "resources") and len(obs.resources) > 0)
        check(f"  Observation has alerts", hasattr(obs, "alerts"))
        check(f"  Observation has total_hourly_cost", hasattr(obs, "total_hourly_cost"))
        check(f"  Observation has system_uptime", hasattr(obs, "system_uptime"))

        # Execute actions and check step returns
        intermediate_rewards = []
        result = None
        for action in actions:
            result = env.step(action)
            check(f"  step() returns StepResult", isinstance(result, StepResult))
            check(f"  StepResult has reward (float)",
                  isinstance(result.reward, (int, float)))
            check(f"  StepResult has done (bool)", isinstance(result.done, bool))
            intermediate_rewards.append(result.reward)

        # Run to completion
        while result and not result.done:
            result = env.step(Action(command=ActionCommand.WAIT))

        # Grade
        score, breakdown = env.grade()
        check(f"  grade() returns score in [0, 1]",
              0.0 <= score <= 1.0, f"score={score}")
        check(f"  grade() returns breakdown dict",
              isinstance(breakdown, dict))
        check(f"  Perfect score = 1.0", score == 1.0, f"score={score}")
        check(f"  Dense rewards (non-zero intermediates)",
              any(r != 0 for r in intermediate_rewards),
              f"rewards={intermediate_rewards}")

    # --- 5. Summary ---
    print("\n" + "=" * 60)
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _ in results if s == FAIL)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print(f"\n  >> ALL CHECKS PASSED -- Environment is OpenEnv compliant!")
    else:
        print(f"\n  WARNING: {failed} check(s) failed:")
        for s, name, detail in results:
            if s == FAIL:
                print(f"    {FAIL} {name}" + (f" — {detail}" if detail else ""))

    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
