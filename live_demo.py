from pathlib import Path

from analysis_plots import save_evaluation_plots
from gymnasium_env import NetworkSecurityEnv
from stable_baselines3 import DQN

BASE_DIR = Path(__file__).resolve().parent


def run_live_demo(num_packets=5000):
    print(f"--- STARTING LIVE IDS DEMONSTRATION ({num_packets} PACKETS) ---")

    test_csv = str(BASE_DIR / "dataset_TEST_clean.csv")
    env = NetworkSecurityEnv(test_csv, is_training=False)

    model_path = str(BASE_DIR / "dqn_network_security_final")
    try:
        model = DQN.load(model_path)
        print("Model loaded successfully.\n")
    except:
        print("Error: Model file not found!")
        return

    # Compteurs pour les métriques finales
    stats = {
        "bon_autorise": 0,  # True Negative
        "mauvais_bloque": 0,  # True Positive
        "bon_bloque": 0,  # False Positive (Erreur)
        "mauvais_autorise": 0,  # False Negative (Danger !)
    }
    agent_actions = []
    ground_truth = []
    rewards = []

    obs, info = env.reset()

    print(f"{'STEP':<8} | {'DÉCISION':<15} | {'VÉRITÉ':<10} | {'RÉSULTAT'}")
    print("-" * 55)

    for i in range(num_packets):
        # Prédiction de l'agent
        action, _ = model.predict(obs, deterministic=True)

        # Récupération du label réel AVANT le step
        true_label = env.labels[env.current_step]
        is_attack = true_label != "BENIGN"
        agent_actions.append(int(action))
        ground_truth.append(1 if is_attack else 0)

        # Logique de comparaison
        if action == 1:  # Bloqué (Honeypot)
            if is_attack:
                res_text = "✅ MAUVAIS BLOQUÉ"
                stats["mauvais_bloque"] += 1
            else:
                res_text = "❌ BON BLOQUÉ (ERREUR)"
                stats["bon_bloque"] += 1
        else:  # Autorisé (Production)
            if not is_attack:
                res_text = "✅ BON AUTORISÉ"
                stats["bon_autorise"] += 1
            else:
                res_text = "🔥 MAUVAIS AUTORISÉ (FUITE!)"
                stats["mauvais_autorise"] += 1

        # Affichage toutes les 10 lignes pour la lisibilité ou avec un petit délai
        if i % 10 == 0 or i < 50:
            decision_str = "HONEYPOT" if action == 1 else "PRODUCTION"
            truth_str = "ATTACK" if is_attack else "SAFE"
            print(f"{i:<8} | {decision_str:<15} | {truth_str:<10} | {res_text}")

        # Exécuter le pas dans l'env
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    # --- BILAN FINAL ---
    print("\n" + "=" * 50)
    print("                BILAN DES MÉTRIQUES                ")
    print("=" * 50)
    print(f"🟢 Bons paquets autorisés (Production) : {stats['bon_autorise']}")
    print(f"🔴 Mauvais bloqués (Honeypot)          : {stats['mauvais_bloque']}")
    print(f"🟠 Bons bloqués par erreur (Faux Pos.) : {stats['bon_bloque']}")
    print(f"💀 Mauvais autorisés (Danger/Fuite)    : {stats['mauvais_autorise']}")
    print("-" * 50)

    total = sum(stats.values())
    accuracy = (stats["bon_autorise"] + stats["mauvais_bloque"]) / total * 100
    print(f"PRÉCISION GLOBALE : {accuracy:.2f}%")
    print("=" * 50)
    save_evaluation_plots(
        ground_truth,
        agent_actions,
        rewards,
        output_prefix="live_demo",
    )
    print(f"Live demo graphs and interpretations exported to: {BASE_DIR / 'results'}")


if __name__ == "__main__":
    run_live_demo(10000)
