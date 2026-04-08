"""
Grader for Adaptive Study Environment
Returns scores for 3 tasks, each strictly in (0, 1).
"""

from env import AdaptiveStudyEnv
from agent import smart_policy


def _clamp(value: float) -> float:
    """Clamp to strictly open interval (0, 1)."""
    return max(0.01, min(0.99, float(value)))


def _run_episode(episodes: int = 10):
    """Run N episodes and return per-episode metrics."""
    results = []
    for _ in range(episodes):
        env = AdaptiveStudyEnv()
        state = env.reset()
        total_reward = 0.0
        final_state = state

        done = False
        while not done:
            action = smart_policy(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            final_state = state

        results.append({
            "total_reward": total_reward,
            "final_mastery": final_state["mastery"],
            "final_stress": final_state["stress"],
            "final_focus": final_state["focus"],
            "final_energy": final_state["energy"],
        })
    return results


def grade_task1(episodes: int = 10) -> float:
    """
    Task 1 — Mastery Achievement
    Score = average final mastery across episodes.
    """
    results = _run_episode(episodes)
    avg_mastery = sum(r["final_mastery"] for r in results) / len(results)
    return _clamp(avg_mastery)


def grade_task2(episodes: int = 10) -> float:
    """
    Task 2 — Stress Control
    Score = 1 - average final stress (lower stress → higher score).
    """
    results = _run_episode(episodes)
    avg_stress = sum(r["final_stress"] for r in results) / len(results)
    score = 1.0 - avg_stress
    return _clamp(score)


def grade_task3(episodes: int = 10) -> float:
    """
    Task 3 — Focus × Energy Balance
    Score = average of (focus + energy) / 2 across episodes.
    """
    results = _run_episode(episodes)
    avg_balance = sum(
        (r["final_focus"] + r["final_energy"]) / 2.0
        for r in results
    ) / len(results)
    return _clamp(avg_balance)


def grade(episodes: int = 10) -> dict:
    """
    Main grading entry-point.
    Returns a dict with scores for all 3 tasks, each strictly in (0, 1).
    """
    scores = {
        "task1": grade_task1(episodes),
        "task2": grade_task2(episodes),
        "task3": grade_task3(episodes),
    }
    print("Grader scores:", scores)
    return scores


if __name__ == "__main__":
    result = grade()
    for task, score in result.items():
        assert 0.0 < score < 1.0, f"{task} score {score} is out of (0, 1)!"
    print("All scores valid:", result)
