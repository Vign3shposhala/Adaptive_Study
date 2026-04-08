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

    focus_gain = state["focus"] - prev_state["focus"]
    mastery_gain = state["mastery"] - prev_state["mastery"]
    stress_penalty = state["stress"]

    raw = 0.4 * focus_gain + 0.5 * mastery_gain - 0.3 * stress_penalty

    reward = 1 / (1 + math.exp(-raw))

    # strict safe range
    reward = max(0.01, min(0.99, reward))

    return reward
