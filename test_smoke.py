"""Quick smoke test for all upgraded modules."""
import sys

def test_tasks():
    from tasks import list_tasks, get_task
    tasks = list_tasks()
    assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
    for t in tasks:
        tc = get_task(t["id"])
        s1 = tc.get_initial_state(seed=42)
        s2 = tc.get_initial_state()
        assert "resources" in s1 and "resources" in s2
    print("[OK] Tasks: 3 tasks with seed support")

def test_env():
    from env import CloudSREEnv
    e = CloudSREEnv()
    obs = e.reset("phantom_volume_cleanup", seed=42)
    assert len(obs.resources) > 0
    assert obs.total_hourly_cost > 0
    
    # Test virtual tools
    cost_report = e.analyze_costs()
    assert "Cost Analysis" in cost_report
    alerts_report = e.check_alerts()
    assert len(alerts_report) > 0
    
    # Test step
    from models import Action, ActionCommand
    result = e.step(Action(command=ActionCommand.INSPECT, resource_id=obs.resources[0].id))
    assert result.reward is not None
    assert len(e.get_cost_history()) == 2
    print(f"[OK] Env: {len(obs.resources)} resources, virtual tools, chaos ready")

def test_react_agent():
    from react_agent import ReActAgent, ReActTrace, ReActStep
    agent = ReActAgent(api_key="test-key", model="gpt-4o-mini")
    
    # Test parsing
    sample = '**Thought**: I see unattached volumes.\n\n**Action**: {"command": "inspect", "resource_id": "ebs-001"}'
    thought, action_json = agent._parse_react_response(sample)
    assert "unattached" in thought
    assert action_json["command"] == "inspect"
    assert action_json["resource_id"] == "ebs-001"
    
    # Test trace formatting
    trace = ReActTrace(task_id="test", model="gpt-4o-mini")
    step = ReActStep(
        step_number=1, thought="Analyzing...", action_text="inspect(ebs-001)",
        action_json={"command": "inspect"}, observation="Details...",
        reward=0.01, cumulative_reward=0.01
    )
    trace.add_step(step)
    md = trace.to_markdown()
    assert "ReAct Agent Trace" in md
    print("[OK] ReAct Agent: parsing & trace generation")

def test_arena():
    from arena import AgentArena
    arena = AgentArena(api_key=None)  # No API key = mock results
    results = arena.run_arena("phantom_volume_cleanup", seed=42, models=[])
    assert len(results) >= 2  # Mock results for Claude, Llama
    html = arena.generate_leaderboard_html()
    assert "Agent Arena" in html
    print(f"[OK] Arena: {len(results)} models, leaderboard HTML generated")

def test_app_imports():
    # Just test that app.py imports cleanly
    from app import generate_metric_cards, generate_topology_html, generate_alerts_html
    from env import CloudSREEnv
    e = CloudSREEnv()
    obs = e.reset("noisy_neighbor_incident", seed=123)
    
    metrics = generate_metric_cards(obs)
    assert "div" in metrics.lower()
    topo = generate_topology_html(obs)
    assert "div" in topo.lower()
    alerts = generate_alerts_html(obs)
    assert "div" in alerts.lower()
    print("[OK] App: dashboard functions render correctly")

if __name__ == "__main__":
    tests = [test_tasks, test_env, test_react_agent, test_arena, test_app_imports]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as ex:
            print(f"[FAIL] {t.__name__}: {ex}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All smoke tests PASSED!")
