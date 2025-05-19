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
    
    # Seed for reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Show config
    print(OmegaConf.to_yaml(cfg))

    # Load data
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        batch_size=cfg.batch_size,
        validation_split=cfg.dataset.validation_split,
        data_dir="./dataset/ptbxl"
    )
    logger.info("✅ PTB-XL data loaded")

    # Load model
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    # Training and testing functions
    train_torch = models.train_torch()
    test_torch = models.test_torch()

    # Load latest model checkpoint if exists
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info("📦 Loading latest local model checkpoint")
        model = client_utils.download_local_model(model_type, task_id, listdir=local_list, model=model)

    # Register everything
    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch
    }

    # Launch FedOps Client
    logger.info("🚀 Starting FL client task")
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
