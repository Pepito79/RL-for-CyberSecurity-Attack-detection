import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES_TO_KEEP = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Fwd IAT Total",
    "Fwd Packets/s",
    "Packet Length Mean",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
]
FINAL_COLUMNS = FEATURES_TO_KEEP + ["Label"]

# On regroupe ABSOLUMENT TOUS les fichiers jours de la semaine pour avoir toutes les attaques
TOUS_LES_FICHIERS = [
    "/home/pepito/Documents/Python/Reddis/RL/Data/Monday-WorkingHours.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Wednesday-workingHours.pcap_ISCX.csv",
]


def generer_datasets_melanges(
    input_files, train_filename, test_filename, test_size=0.20
):
    df_list = []
    for f in input_files:
        if not os.path.exists(f):
            print(f"Warning: Fichier {f} non trouvé. Passé.")
            continue
        print(f"Lecture de : {f}...")
        for chunk in pd.read_csv(f, chunksize=50000):
            chunk.columns = chunk.columns.str.strip()
            df_list.append(chunk[FINAL_COLUMNS])

    if not df_list:
        print("Erreur: Aucun fichier valide trouvé.")
        return

    print("\nFusion de tous les fichiers en un seul dataset global...")
    df_global = pd.concat(df_list, ignore_index=True)

    print("Nettoyage des valeurs infinies (Inf) et manquantes (NaN)...")
    df_global = df_global.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(
        f"Séparation aléatoire globale (Train: {int((1 - test_size) * 100)}% / Test: {int(test_size * 100)}%)..."
    )
    # train_test_split mélange automatiquement les données avant de couper (shuffle=True par défaut)
    df_train, df_test = train_test_split(
        df_global, test_size=test_size, random_state=42
    )

    # Réinitialisation des index
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Sauvegarde Train
    train_path = f"/home/pepito/Documents/Python/Reddis/RL/{train_filename}"
    print(
        f"Sauvegarde du dataset d'ENTRAÎNEMENT : {train_path} ({len(df_train)} lignes)"
    )
    df_train.to_csv(train_path, index=False)

    # Sauvegarde Test
    test_path = f"/home/pepito/Documents/Python/Reddis/RL/{test_filename}"
    print(f"Sauvegarde du dataset de TEST : {test_path} ({len(df_test)} lignes)")
    df_test.to_csv(test_path, index=False)


if __name__ == "__main__":
    print("--- STARTING GLOBAL MIXED PREPROCESSING ---\n")
    generer_datasets_melanges(
        input_files=TOUS_LES_FICHIERS,
        train_filename="dataset_TRAIN_clean.csv",
        test_filename="dataset_TEST_clean.csv",
        test_size=0.20,  # 20% des données globales réservées strictement au test
    )
    print("\n--- PREPROCESSING COMPLETED ---")
