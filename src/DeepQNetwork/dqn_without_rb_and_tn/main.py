import gymnasium as gym
from .agent import Agent
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    env = gym.make('LunarLander-v3')

    input_features = env.observation_space.shape[0]
    intermediate_dim = 128
    lr = 1e-3
    gamma = 0.99
    eps = 1.0
    eps_dec = 1e-4
    eps_min = 0.01
    n_games = 10000

    scores = []
    eps_history = []
    agent = Agent(env,input_features,intermediate_dim,lr,gamma,eps,eps_dec,eps_min)

    for episode in tqdm(range(n_games)):

        score = 0
        done = False
        obs,_ = env.reset()

        while not done:

            action = agent.choose_action(obs)
            obs_,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.learn(obs,action,reward,obs_)
            obs = obs_

        scores.append(score)
        eps_history.append(agent.eps)

        if (episode + 1) % 100 == 0:

            avg_score = np.mean(scores[-100:])
            print(
                f"Episode: {episode + 1} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Epsilon: {agent.eps:.3f}"
            )

    env.close()    






