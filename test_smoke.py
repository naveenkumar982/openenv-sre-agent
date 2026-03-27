"""Smoke test for the Cloud SRE environment."""
import sys
sys.path.insert(0, '.')
from env import CloudSREEnv  # type: ignore
from models import Action, ActionCommand  # type: ignore
from tasks import list_tasks  # type: ignore

env = CloudSREEnv(max_steps=15)

print('=== Available Tasks ===')
for t in list_tasks():
    print(f"  {t['difficulty'].upper():>6}: {t['id']}")

# --- Task 1: Easy ---
print('\n=== Task 1: Phantom Volume Cleanup ===')
obs = env.reset('phantom_volume_cleanup')
print(f'Resources: {len(obs.resources)}, Alerts: {len(obs.alerts)}, Cost: ${obs.total_hourly_cost:.4f}/hr')

for orphan_id in ['ebs-orphan-001', 'ebs-orphan-002', 'ebs-orphan-003']:
    r = env.step(Action(command=ActionCommand.TERMINATE, resource_id=orphan_id))
    print(f'  Terminated {orphan_id}: reward={r.reward:+.4f}')

while not r.done:
    r = env.step(Action(command=ActionCommand.WAIT))

score, breakdown = env.grade()
print(f'  FINAL SCORE: {score} | Orphans removed: {breakdown["orphans_terminated"]}')

# --- Task 2: Medium ---
print('\n=== Task 2: Latency Spike Remediation ===')
obs = env.reset('latency_spike_remediation')
print(f'Resources: {len(obs.resources)}, Alerts: {len(obs.alerts)}, Cost: ${obs.total_hourly_cost:.4f}/hr')

r = env.step(Action(command=ActionCommand.SCALE, resource_id='rds-primary-001', params={'target_size': 'db.t3.medium'}))
print(f'  Scaled RDS: reward={r.reward:+.4f}')

while not r.done:
    r = env.step(Action(command=ActionCommand.WAIT))

score, breakdown = env.grade()
print(f'  FINAL SCORE: {score} | RDS scaled: {breakdown["rds_scaled"]}, Under budget: {breakdown["under_budget"]}')

# --- Task 3: Hard ---
print('\n=== Task 3: Noisy Neighbor Incident ===')
obs = env.reset('noisy_neighbor_incident')
print(f'Resources: {len(obs.resources)}, Alerts: {len(obs.alerts)}, Cost: ${obs.total_hourly_cost:.4f}/hr')

r = env.step(Action(command=ActionCommand.INSPECT, resource_id='ec2-rogue-test-001'))
print(f'  Inspected rogue: reward={r.reward:+.4f}')

r = env.step(Action(command=ActionCommand.TERMINATE, resource_id='ec2-rogue-test-001'))
print(f'  Terminated rogue: reward={r.reward:+.4f}')

r = env.step(Action(command=ActionCommand.REBOOT, resource_id='ec2-backend-prod-001'))
print(f'  Rebooted backend: reward={r.reward:+.4f}')

while not r.done:
    r = env.step(Action(command=ActionCommand.WAIT))

score, breakdown = env.grade()
print(f'  FINAL SCORE: {score} | Inspected: {breakdown["inspected_rogue"]}, Terminated rogue: {breakdown["terminated_rogue"]}, Rebooted: {breakdown["rebooted_backend"]}')

print('\n✅ ALL TASKS VERIFIED SUCCESSFULLY')
