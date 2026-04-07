from env import AdaptiveStudyEnv
from agent import smart_policy

def evaluate(episodes=10):
    scores = []

    for _ in range(episodes):
        env = AdaptiveStudyEnv()
        state = env.reset()
        total = 0

        done = False
        while not done:
            action = smart_policy(state)
            state, reward, done, _ = env.step(action)
            total += reward

        score = max(0, min(1, total / 1000))
        scores.append(score)

    print("Average Score:", sum(scores)/len(scores))


if __name__ == "__main__":
    evaluate()