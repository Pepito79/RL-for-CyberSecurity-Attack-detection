import os

import numpy as np
import pandas as pd

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

FICHIERS_TRAIN = [
    "/home/pepito/Documents/Python/Reddis/RL/Data/Monday-WorkingHours.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "/home/pepito/Documents/Python/Reddis/RL/Data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]
FICHIER_TEST = [
    "/home/pepito/Documents/Python/Reddis/RL/Data/Wednesday-workingHours.pcap_ISCX.csv"
]


def clean_and_save_dataset(input_files, output_filename):
    df_list = []
    for f in input_files:
        if not os.path.exists(f):
            print(f"Warning: File {f} not found. Skipping.")
            continue
        print(f"Processing: {f}...")
        for chunk in pd.read_csv(f, chunksize=50000):
            chunk.columns = chunk.columns.str.strip()
            df_list.append(chunk[FINAL_COLUMNS])

    if not df_list:
        print(f"Error: No valid source data found for {output_filename}.")
        return

    print("Concatenating data frames...")
    df_final = pd.concat(df_list, ignore_index=True)

    print("Cleaning infinite (Inf) and missing (NaN) values...")
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Apply global random shuffling only to the training split to optimize the Replay Buffer
    if "TRAIN" in output_filename:
        print(
            "Shuffling training dataset rows to interleave clean and threat signatures..."
        )
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = f"/home/pepito/Documents/Python/Reddis/RL/{output_filename}"
    print(f"Saving filtered dataset to: {output_path}...")
    df_final.to_csv(output_path, index=False)
    print(f"Saved {output_filename} successfully! Total Rows: {len(df_final)}\n")


if __name__ == "__main__":
    print("--- STARTING CIC-IDS2017 DATASET PREPROCESSING ---\n")
    clean_and_save_dataset(FICHIERS_TRAIN, "dataset_TRAIN_clean.csv")
    clean_and_save_dataset(FICHIER_TEST, "dataset_TEST_clean.csv")
    print("--- PREPROCESSING COMPLETED ---")
