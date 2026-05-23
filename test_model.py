from gymnasium_env import NetworkSecurityEnv
from sklearn.metrics import classification_report, confusion_matrix
from stable_baselines3 import DQN


def test_agent():
    print("--- DQN MODEL EVALUATION PHASE ---")

    # Load environment flagged for evaluation (uses preserved scaling parameters)
    test_csv = "/home/pepito/Documents/Python/Reddis/RL/dataset_TEST_clean.csv"
    env = NetworkSecurityEnv(test_csv, is_training=False)

    print("Loading trained deep neural policy framework...")
    model = DQN.load(
        "/home/pepito/Documents/Python/Reddis/RL/dqn_network_security_final"
    )

    agent_actions = []
    ground_truth = []

    obs, info = env.reset()
    done = False

    print("Evaluating streaming telemetry profiles against policy space...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        agent_actions.append(action)

        true_label = env.labels[env.current_step]
        ground_truth.append(0 if true_label == "BENIGN" else 1)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Evaluation engine process tracking complete.\n")
    analyze_results(ground_truth, agent_actions)


def analyze_results(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 40)
    print("          CONFUSION MATRIX          ")
    print("=" * 40)
    print(f"True Negatives  (Valid Traffic to Prod) : {tn}")
    print(f"False Positives (Valid Traffic to Honey): {fp}")
    print(f"False Negatives (Threat Leaked to Prod)  : {fn}")
    print(f"True Positives  (Threat Isolated)        : {tp}")
    print("=" * 40)

    print("\n" + "=" * 40)
    print("        CLASSIFICATION REPORT       ")
    print("=" * 40)
    print(
        classification_report(
            y_true, y_pred, target_names=["Production (Valid)", "Honeypot (Threat)"]
        )
    )
    print("=" * 40)


if __name__ == "__main__":
    test_agent()
