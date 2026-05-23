import torch
from gymnasium_env import NetworkSecurityEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training target hardware execution node: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Load environment explicitly flagged for training
    train_csv = "/home/pepito/Documents/Python/Reddis/RL/dataset_TRAIN_clean.csv"
    env = NetworkSecurityEnv(train_csv, is_training=True)
    env = Monitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="/home/pepito/Documents/Python/Reddis/RL/models/",
        name_prefix="dqn_ids_model",
    )

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
        tensorboard_log="/home/pepito/Documents/Python/Reddis/RL/tensorboard_logs/",
    )

    print("Executing model optimization loop...")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
    print("Training loop finished successfully.")

    model.save("/home/pepito/Documents/Python/Reddis/RL/dqn_network_security_final")
    print("Final deep policy model configuration exported to storage.")


if __name__ == "__main__":
    train_agent()
