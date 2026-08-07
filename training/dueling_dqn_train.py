import numpy as np
import torch
from src.dueling_dqn.agent import Agent
from tqdm import tqdm
import gymnasium as gym
import yaml
from utils.training_curve_plot import plot_training_curves

with open("configs/dqn_config.yaml", "r") as f:

    config = yaml.safe_load(f)

def trainer(config):

    env =  gym.make(config["environment"]["name"],render_mode=config["environment"]["render_mode"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = Agent(max_memories=config["train"]["max_memories"],
                  discount_factor=config["train"]["discount_factor"], 
                  learning_rate=config["train"]["learning_rate"],
                  num_state_features=config["train"]["num_state_features"], 
                  num_actions=config["train"]["num_actions"],
                  intermediate_dims=config["train"]["intermediate_dims"],
                  epsilon=config["train"]["epsilon"], 
                  epsilon_decay=config["train"]["epsilon_decay"], 
                  min_epsilon=config["train"]["min_epsilon"],
                  device=device)

    ending_tol = 0
    log = {"scores": [],"running_avg_scores": []}

    for i in tqdm(range(config["train"]["num_games"])):

        score = 0
        step = 0
        state, _ = env.reset()
        done = False

        while not done:

            action = agent.select_action(state)
            next_state, reward, terminal, truncated, _= env.step(action)
            done = terminal or truncated
            score += reward
            agent.replay_buffer.update_memory(state,next_state,action,reward,done)
            agent.train_step(config["train"]["batch_size"])

            if step % config["train"]["update_target_freq"] == 0:

                agent.update_target_network()

            step += 1
            state = next_state

        log["scores"].append(score)
        running_avg_score = np.mean(log["scores"][-config["train"]["running_avg_steps"]:])
        log["running_avg_scores"].append(running_avg_score)
        
        if i % config["train"]["log_freq"] == 0:

            tqdm.write(f"Game #: {i} | Score: {score} | Moving Avg Scores: {running_avg_score} | Epsilon: {agent.epsilon}")
        
        if score >= config["train"]["min_reward"]:

            ending_tol += 1

            if ending_tol == config["train"]["game_tolerance"]:

                break
        else:

            ending_tol = 0
            
    print("Completed Training")

    return agent,log
   
if __name__ == "__main__":

    agent, log = trainer(config)
    plot_training_curves(log)