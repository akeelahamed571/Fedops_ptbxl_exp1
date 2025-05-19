import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import wfdb
import time


def download_ptbxl_if_needed(download_dir="./dataset/ptbxl"):
    zip_path = os.path.join(download_dir, "ptbxl.zip")
    extract_dir = os.path.join(download_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    csv_path = os.path.join(extract_dir, "ptbxl_database.csv")

    if os.path.exists(csv_path):
        print("✅ PTB-XL dataset already exists. Skipping download.")
        return

    os.makedirs(download_dir, exist_ok=True)
    print("📥 Downloading PTB-XL dataset...")
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip"
    urllib.request.urlretrieve(url, zip_path)
    print("✅ Download complete.")

    print("📦 Extracting PTB-XL dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("✅ Extraction complete.")


class PTBXLDataset(Dataset):
    def __init__(self, df, data_path, leads):
        print(f"📊 Initializing dataset with {len(df)} samples")
        self.df = df
        self.data_path = data_path
        self.leads = leads
        self.mlb = MultiLabelBinarizer()
        print("🔄 Binarizing SCP codes into multi-label vectors...")
        self.labels = self.mlb.fit_transform(self.df['scp_codes'].map(eval).map(lambda x: list(x.keys())))
        print("✅ Labels transformed.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = os.path.join(self.data_path, row['filename_lr'].replace('.dat', ''))
        signals, _ = wfdb.rdsamp(record_path)
        signals = signals.T
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(signals, dtype=torch.float32), label


def load_partition(data_dir="./dataset/ptbxl", batch_size=32, validation_split=0.2):
    print("🧩 Step 1: Ensure dataset is downloaded...")
    download_ptbxl_if_needed(data_dir)

    extract_dir = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    print("📂 Step 2: Loading metadata CSV...")
    df = pd.read_csv(os.path.join(extract_dir, "ptbxl_database.csv"))
    print(f"🧾 Total records loaded: {len(df)}")

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    print("🧪 Step 3: Splitting into train/val/test...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=42)
    print(f"📊 Train size: {len(train_df)} | Validation size: {len(val_df)} | Test size: {len(test_df)}")

    print("🧠 Step 4: Creating PyTorch data loaders...")
    train_loader = DataLoader(PTBXLDataset(train_df, extract_dir, leads), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PTBXLDataset(val_df, extract_dir, leads), batch_size=batch_size)
    test_loader = DataLoader(PTBXLDataset(test_df, extract_dir, leads), batch_size=batch_size)
    print("✅ Data loaders ready.")

    return train_loader, val_loader, test_loader


def gl_model_torch_validation(batch_size):
    print("🔍 Preparing global validation loader...")
    _, val_loader, _ = load_partition(batch_size=batch_size)
    return val_loader
