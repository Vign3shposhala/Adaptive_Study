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


def compute_reward(prev_state, new_state):
    reward = 0

    reward += (new_state["mastery"] - prev_state["mastery"]) * 100
    reward += (new_state["focus"] - prev_state["focus"]) * 50
    reward -= new_state["stress"] * 10

    if new_state["energy"] < 0.2:
        reward -= 15

    return reward