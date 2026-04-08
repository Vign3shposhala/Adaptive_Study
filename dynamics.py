def apply_action(state, action):
    if action == "EASY":
        state["mastery"] += 0.04
        state["focus"] += 0.02

    elif action == "HARD":
        if state["focus"] > 0.6:
            state["mastery"] += 0.1
        else:
            state["stress"] += 0.12

    elif action == "VIDEO":
        state["focus"] += 0.05
        state["energy"] -= 0.05

    elif action == "BREAK":
        state["energy"] += 0.12
        state["stress"] -= 0.1

    elif action == "REVISE":
        state["mastery"] += 0.07

    elif action == "MOTIVATE":
        state["focus"] += 0.1
        state["stress"] -= 0.05

    return state


def compute_reward(prev_state, state):
    import math

    # stable features
    focus = state["focus"]
    mastery = state["mastery"]
    stress = state["stress"]

    # bounded linear combination
    raw = 0.4 * focus + 0.5 * mastery - 0.3 * stress

    # sigmoid → guarantees (0,1)
    reward = 1 / (1 + math.exp(-raw))

    # 🔥 CRITICAL: avoid edges COMPLETELY
    if reward <= 0:
        reward = 0.01
    elif reward >= 1:
        reward = 0.99

    return reward
