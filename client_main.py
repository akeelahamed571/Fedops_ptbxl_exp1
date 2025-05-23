import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models
from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Logging setup
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("🔧 Step 1: Setting random seeds")
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print("📋 Step 2: Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("📦 Step 3: Loading PTB-XL dataset...")
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        batch_size=cfg.batch_size,
        validation_split=cfg.dataset.validation_split,
        data_dir="./dataset/ptbxl"
    )
    logger.info("✅ Step 3 Done: PTB-XL data loaded")

    print("🧠 Step 4: Instantiating model...")
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    logger.info(f"✅ Model '{model_name}' instantiated")

    print("🛠️ Step 5: Getting training and testing functions...")
    train_torch = models.train_torch()
    test_torch = models.test_torch()
    logger.info("✅ Step 5 Done: Training and test functions ready")

    print("📁 Step 6: Checking for existing local model checkpoint...")
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info("📦 Step 6 Done: Loading local model checkpoint")
        model = client_utils.download_local_model(model_type, task_id, listdir=local_list, model=model)
    else:
        logger.info("📁 No local model found. Starting fresh.")

    print("📝 Step 7: Registering model and data with FL client...")
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch
    }

    print("🚀 Step 8: Launching FL client task...")
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()
    logger.info("🏁 Training started!")


if __name__ == "__main__":
    main()
