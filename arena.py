"""
Agent Arena — Multi-model comparison and leaderboard for Cloud SRE OpenEnv.

Runs the same seeded scenario with multiple AI agents and compares results.
Supports GPT-4o, GPT-4o-mini, and mock results for models without API keys.

Usage:
    arena = AgentArena(api_key="sk-...")
    results = arena.run_arena("noisy_neighbor_incident", seed=42)
    html = arena.generate_leaderboard_html()
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ArenaResult:
    """Result for a single model run in the arena."""
    model_name: str
    display_name: str
    score: float
    steps_used: int
    cost_saved: float
    reasoning_quality: str  # "high", "medium", "low"
    time_taken: float
    is_mock: bool = False


class AgentArena:
    """
    Runs multiple AI agents against the same seeded scenario
    and produces a leaderboard comparison.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.results: List[ArenaResult] = []
        self._last_task_id: Optional[str] = None

    def run_arena(self, task_id: str, seed: int = 42,
                  models: Optional[List[str]] = None) -> List[ArenaResult]:
        """
        Run all available models on the same seeded scenario.

        Args:
            task_id: Task to benchmark
            seed: RNG seed (ensures identical scenarios)
            models: Optional list of model names. Defaults to
                     ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

        Returns:
            List of ArenaResult sorted by score (descending)
        """
        from env import CloudSREEnv
        from react_agent import ReActAgent

        if models is None:
            models = ["gpt-4o-mini", "gpt-4o"]

        self.results = []
        self._last_task_id = task_id

        for model_name in models:
            display = self._display_name(model_name)

            if not self.api_key:
                # No API key — use mock results
                self.results.append(self._mock_result(model_name, display))
                continue

            try:
                env = CloudSREEnv(max_steps=15)
                agent = ReActAgent(api_key=self.api_key, model=model_name)
                trace = agent.run_episode(env, task_id, seed=seed)

                quality = "high"
                if trace.final_score < 0.5:
                    quality = "low"
                elif trace.final_score < 0.8:
                    quality = "medium"

                self.results.append(ArenaResult(
                    model_name=model_name,
                    display_name=display,
                    score=trace.final_score,
                    steps_used=len(trace.steps),
                    cost_saved=round(trace.initial_cost - trace.final_cost, 4),
                    reasoning_quality=quality,
                    time_taken=round(trace.total_time, 1),
                    is_mock=False,
                ))
            except Exception as e:
                # If API call fails, add mock result
                self.results.append(ArenaResult(
                    model_name=model_name,
                    display_name=display,
                    score=0.0,
                    steps_used=0,
                    cost_saved=0.0,
                    reasoning_quality="error",
                    time_taken=0.0,
                    is_mock=True,
                ))

        # Add mock results for models we can't call
        mock_models = {
            "claude-sonnet": ("Claude Sonnet 4", 0.92, 5, 0.68, "high", 8.3),
            "llama-3.1-70b": ("Llama 3.1 70B", 0.78, 7, 0.45, "medium", 12.1),
        }
        for mname, (dname, score, steps, saved, qual, ttime) in mock_models.items():
            if mname not in [r.model_name for r in self.results]:
                self.results.append(ArenaResult(
                    model_name=mname, display_name=dname,
                    score=score, steps_used=steps, cost_saved=saved,
                    reasoning_quality=qual, time_taken=ttime, is_mock=True,
                ))

        # Sort by score descending
        self.results.sort(key=lambda r: r.score, reverse=True)
        return self.results

    def generate_leaderboard_html(self) -> str:
        """Generate a visually rich HTML leaderboard."""
        if not self.results:
            return '<div class="arena-empty">No arena results yet. Run a comparison first.</div>'

        medals = ["🥇", "🥈", "🥉"]
        max_score = max(r.score for r in self.results) if self.results else 1.0
        if max_score == 0:
            max_score = 1.0

        rows = []
        for i, r in enumerate(self.results):
            medal = medals[i] if i < 3 else f"#{i+1}"
            bar_pct = int((r.score / max_score) * 100) if max_score > 0 else 0
            quality_badge = {
                "high": '<span class="q-badge q-high">High</span>',
                "medium": '<span class="q-badge q-med">Medium</span>',
                "low": '<span class="q-badge q-low">Low</span>',
                "error": '<span class="q-badge q-err">Error</span>',
            }.get(r.reasoning_quality, "")
            mock_tag = ' <span class="mock-tag">(simulated)</span>' if r.is_mock else ' <span class="live-tag">LIVE</span>'

            rows.append(f"""
            <div class="lb-row" style="animation-delay: {i * 0.1}s">
                <div class="lb-rank">{medal}</div>
                <div class="lb-info">
                    <div class="lb-name">{r.display_name}{mock_tag}</div>
                    <div class="lb-bar-container">
                        <div class="lb-bar" style="width: {bar_pct}%"></div>
                    </div>
                </div>
                <div class="lb-stats">
                    <div class="lb-score">{r.score:.2f}</div>
                    <div class="lb-meta">{r.steps_used} steps &middot; {r.time_taken:.1f}s</div>
                    <div class="lb-meta">Saved ${r.cost_saved:.2f}/hr {quality_badge}</div>
                </div>
            </div>
            """)

        task_display = self._last_task_id or "Unknown"

        return f"""
        <div class="arena-container">
            <div class="arena-header">
                <h3>🏟 Agent Arena — Live Leaderboard</h3>
                <div class="arena-task">Task: {task_display} &middot; Same seed for fair comparison</div>
            </div>
            {"".join(rows)}
        </div>
        """

    def _display_name(self, model: str) -> str:
        names = {
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "claude-sonnet": "Claude Sonnet 4",
            "llama-3.1-70b": "Llama 3.1 70B",
        }
        return names.get(model, model)

    def _mock_result(self, model: str, display: str) -> ArenaResult:
        """Generate plausible mock results for a model."""
        import random
        rng = random.Random(hash(model))
        return ArenaResult(
            model_name=model, display_name=display,
            score=round(rng.uniform(0.6, 0.95), 2),
            steps_used=rng.randint(4, 10),
            cost_saved=round(rng.uniform(0.2, 1.5), 2),
            reasoning_quality=rng.choice(["high", "medium"]),
            time_taken=round(rng.uniform(5, 20), 1),
            is_mock=True,
        )
