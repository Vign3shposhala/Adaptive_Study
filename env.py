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

        for key in ["focus", "energy", "mastery", "stress"]:
            self.state[key] = max(0, min(1, self.state[key]))

        self.state["time"] += 1

        reward = compute_reward(prev_state, self.state)
        done = self.state["time"] >= self.max_time

        return self._get_obs(), reward, done, {}

    def state(self):
        return self._get_obs()

    def _get_obs(self):
        return self.state.copy()