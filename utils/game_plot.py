from pathlib import Path
import torch
import gymnasium as gym
from IPython.display import Video

def record_game(
    env: gym.Env,
    agent,
    prefix: str = "agent",
    video_dir: str = "savevideos",
    seed: int = 42,
):

    Path(video_dir).mkdir(parents=True, exist_ok=True)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=prefix,
        disable_logger=True,
    )

    state, _ = env.reset(seed=seed)
    done = False

    while not done:

        state_tensor = torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)
        action = agent.inference(state_tensor)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

    return Path(video_dir) / f"{prefix}-episode-0.mp4"