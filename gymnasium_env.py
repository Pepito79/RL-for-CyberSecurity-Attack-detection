import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler


class NetworkSecurityEnv(gym.Env):
    def __init__(
        self,
        csv_path,
        is_training=True,
        scaler_path="/home/pepito/Documents/Python/Reddis/RL/network_scaler.pkl",
    ):
        super(NetworkSecurityEnv, self).__init__()

        self.df = pd.read_csv(csv_path)
        self.labels = self.df["Label"].values
        raw_features = self.df.drop(columns=["Label"]).values

        if is_training:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(raw_features)
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler successfully fitted and saved to: {scaler_path}")
        else:
            try:
                self.scaler = joblib.load(scaler_path)
                self.features = self.scaler.transform(raw_features)
                print(f"Loaded existing scaling profile from: {scaler_path}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing scaler profile file at {scaler_path}. Run training first."
                )

        self.action_space = spaces.Discrete(2)
        num_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.features[self.current_step].astype(np.float32)
        return obs, {}

    def step(self, action):
        true_label = self.labels[self.current_step]

        # Heavily penalize threat leakage to break the collapsed policy
        if true_label == "BENIGN":
            reward = (
                1.0 if action == 0 else -2.0
            )  # Slightly higher penalty for false alarms
        else:
            reward = (
                5.0 if action == 1 else -20.0
            )  # Massive penalty for leaking an active attack
        self.current_step += 1
        terminated = self.current_step >= len(self.df)
        truncated = False

        if not terminated:
            next_obs = self.features[self.current_step].astype(np.float32)
        else:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return next_obs, reward, terminated, truncated, {}
