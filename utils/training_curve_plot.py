import matplotlib.pyplot as plt
import os

def plot_training_curves(log,filename:str,save_dir:str="results_plot",save:bool=True,grid:bool=True) -> None:

    os.makedirs(save_dir,exist_ok=True)

    fig,ax = plt.subplots(figsize=(10, 5))
    ax.plot(log["scores"],label="Scores")
    ax.plot(log["running_avg_scores"],label="Moving Avg")

    ax.set(
        title="Training Scores",
        xlabel="Episode",
        ylabel="Score"
    )

    ax.legend()
    ax.grid(grid)
    fig.tight_layout()

    save_path = os.path.join(save_dir,filename)

    if save:

        fig.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.show()
    plt.close(fig)
