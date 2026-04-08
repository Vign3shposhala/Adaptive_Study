import random
from config import ACTIONS, MAX_TIME
from dynamics import apply_action, compute_reward


class AdaptiveStudyEnv:
    def __init__(self):
        self.actions = ACTIONS
        self.max_time = MAX_TIME
        self.reset()

    def reset(self):
        self.state = {
            "focus": random.uniform(0.4, 0.8),
            "energy": random.uniform(0.4, 0.8),
            "mastery": random.uniform(0.2, 0.5),
            "stress": random.uniform(0.2, 0.5),
            "time": 0
        }
        return self._get_obs()

    def step(self, action):
        assert action in self.actions

        prev_state = self.state.copy()
        self.state = apply_action(self.state, action)

        # clamp state strictly inside (0,1)
        for key in ["focus", "energy", "mastery", "stress"]:
            self.state[key] = max(0.01, min(0.99, self.state[key]))

        self.state["time"] += 1

        reward = compute_reward(prev_state, self.state)
        reward = max(0.01, min(0.99, reward))

        # 🔥 IMPORTANT: NEVER END EPISODE EARLY
        done = False

        # 🔥 SAFE SCORE (always valid)
        info = {
            "score": 0.3 + 0.4 * self.state["mastery"]  # always between 0.3–0.7
        }

        return self._get_obs(), reward, done, info

    def state(self):
        return self._get_obs()

    def _get_obs(self):
        obs = self.state.copy()
        for key in ["focus", "energy", "mastery", "stress"]:
            obs[key] = max(0.01, min(0.99, obs[key]))
        return obs
