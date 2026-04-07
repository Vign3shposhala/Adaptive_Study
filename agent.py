def smart_policy(state):
    if state["stress"] > 0.7:
        return "BREAK"
    if state["energy"] < 0.3:
        return "BREAK"
    if state["focus"] < 0.4:
        return "MOTIVATE"
    if state["mastery"] < 0.5:
        return "EASY"
    return "HARD"