import numpy as np
import torch
from pathlib import Path
from src.DeepQNetwork.dqn.Agent import Agent
from tqdm import tqdm
import gymnasium as gym
import yaml
from utils.game_plot import record_game
from utils.training_curve_plot import plot_training_curves
from utils.wandb_logging import WandbLogger
from utils.profiler import PyTorchProfiler

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
    
    logger = WandbLogger(project=config["logging"]["wandb_project"],
                          run_name=config["logging"].get("run_name"),
                          config=config,
                          mode=config["logging"].get("wandb_mode", "online"),
                          enabled=config["logging"].get("wandb_enabled", True))
    
    logger.watch(agent.dqn, log="gradients", log_freq=100)

    profiler = PyTorchProfiler(log_dir=config["logging"]["profiler_log_dir"],
                                wait=config["profiler"]["wait"],warmup=config["profiler"]["warmup"],
                                active=config["profiler"]["active"],
                                repeat=config["profiler"]["repeat"],
                                export_mode=config["profiler"]["export_mode"])
    

    ending_tol = 0
    log = {"scores": [],"running_avg_scores": []}
    global_step = 0

    with profiler:

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
                profiler.step()

                if step % config["train"]["update_target_freq"] == 0:

                    agent.update_target_network()

                step += 1
                state = next_state
                global_step += 1

            log["scores"].append(score)
            running_avg_score = np.mean(log["scores"][-config["train"]["running_avg_steps"]:])
            log["running_avg_scores"].append(running_avg_score)

            logger.log({"score":score,
                        "running_avg_score":running_avg_score,
                        "epsilon":agent.epsilon,
                        "episode_length": step},step=i)
            
            if i % config["train"]["log_freq"] == 0:

                tqdm.write(f"Game #: {i} | Score: {score} | Moving Avg Scores: {running_avg_score} | Epsilon: {agent.epsilon}")
            
            if score >= config["train"]["min_reward"]:

                ending_tol += 1

                if ending_tol == config["train"]["game_tolerance"]:

                    break
            else:

                ending_tol = 0

    profiler.summary()
    trace_name = f"{config['logging']['run_name']}_trace.json"
    profiler.export_chrome_trace(str(Path(config["logging"]["profiler_log_dir"]) / trace_name))

    video_path = record_game(env,agent,prefix ="dqn_agent",video_dir ="savevideos",seed=42)
    logger.log_video(video_path,name="final_agent_gameplay")

    logger.finish()            
            
    print("Completed Training")

    return agent,log
   
if __name__ == "__main__":

    agent, log = trainer(config)
    plot_training_curves(log,"dqn_agent")


