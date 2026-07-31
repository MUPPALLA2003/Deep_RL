import matplotlib.pyplot as plt

def plot_training_curves(log):

    plt.figure(figsize=(10, 5))
    plt.plot(log["scores"], label="Scores")
    plt.plot(log["running_avg_scores"], label="Moving Avg")
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()