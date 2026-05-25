from pathlib import Path

import torch
from analysis_plots import save_training_plots
from gymnasium_env import NetworkSecurityEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

BASE_DIR = Path(__file__).resolve().parent


class TrainingMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.actions = []

    def _on_step(self):
        rewards = self.locals.get("rewards")
        actions = self.locals.get("actions")

        if rewards is not None:
            self.rewards.extend(float(reward) for reward in rewards)
        if actions is not None:
            self.actions.extend(int(action) for action in actions)

        return True


def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training target hardware execution node: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    train_csv = str(BASE_DIR / "dataset_TRAIN_clean.csv")
    env = NetworkSecurityEnv(train_csv, is_training=True)
    log_file = str(BASE_DIR / "training_stats.csv")

    # On passe le nom du fichier au Monitor
    env = Monitor(env, log_file)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(BASE_DIR / "models"),
        name_prefix="dqn_ids_model",
    )
    metrics_callback = TrainingMetricsCallback()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Lowered slightly to stabilize policy shift
        buffer_size=100000,
        learning_starts=5000,
        batch_size=256,  # Increased to see more diverse samples per batch
        train_freq=1,  # Optimize after every single step
        gradient_steps=1,  # Run gradient updates continuously
        tau=0.005,
        gamma=0.99,
        exploration_fraction=0.20,  # Increased exploration to force it to try action 1 longer
        exploration_final_eps=0.05,
        device=device,
        verbose=1,
        tensorboard_log=str(BASE_DIR / "tensorboard_logs"),
    )

    print("Executing model optimization loop...")
    model.learn(
        total_timesteps=200000,
        callback=[checkpoint_callback, metrics_callback],
        log_interval=1,
    )
    print("Training loop finished successfully.")

    model.save(str(BASE_DIR / "dqn_network_security_final"))
    print("Final deep policy model configuration exported to storage.")

    save_training_plots(metrics_callback.rewards, metrics_callback.actions)
    print(f"Training graphs and interpretations exported to: {BASE_DIR / 'results'}")


if __name__ == "__main__":
    train_agent()
