#!/usr/bin/env python3

import logging
import os
import sys
import traceback
import torch
import cv2
import yaml
import tqdm
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)

# ✅ Global model variable (Singleton)
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads the inpainting model once and keeps it in memory."""
    global model
    if model is None:
        LOGGER.info("🚀 Loading model...")
        model_path = "/app/big-lama"
        checkpoint_path = f"{model_path}/fine-tuned_lama.ckpt"

        # Load config
        train_config_path = os.path.join(model_path, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.load(f)

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        # Load model once
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
        model.to(device)
        model.freeze()
        LOGGER.info("✅ Model loaded successfully.")

def predict_image(image_path, mask_path, output_path="/app/outputs/result.png"):
    """Runs prediction on a single image using the loaded model."""
    try:
        # Ensure model is loaded
        load_model()

        # ✅ Prepare input
        batch = _prepare_input(image_path, mask_path)

        # ✅ Run inference
        with torch.no_grad():
            batch = model(batch)
            cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        # ✅ Save output image
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cur_res)

        return output_path
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        return None

def _prepare_input(image_path, mask_path):
    """Prepares input image and mask as tensors."""
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    mask = torch.tensor(mask).unsqueeze(0).float()

    batch = default_collate([{"image": image, "mask": mask}])
    batch = move_to_device(batch, device)
    batch["mask"] = ((batch["mask"] > 0)).float().to(device)

    return batch
