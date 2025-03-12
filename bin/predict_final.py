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

# ✅ Singleton Model Class (Loads Model Once)
class LaMaModel:
    _instance = None

    def __new__(cls, model_path="/app/big-lama", checkpoint="fine-tuned_lama.ckpt"):
        if cls._instance is None:
            cls._instance = super(LaMaModel, cls).__new__(cls)
            cls._instance._load_model(model_path, checkpoint)
        return cls._instance

    def _load_model(self, model_path, checkpoint):
        """Loads the model ONCE and stores it for reuse."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.checkpoint_path = os.path.join(model_path, checkpoint)

        # Load config correctly
        train_config_path = os.path.join(model_path, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        # Load model ONCE
        self.model = load_checkpoint(train_config, self.checkpoint_path, strict=False, map_location=self.device)
        self.model.to(self.device)
        self.model.freeze()
        LOGGER.info("✅ LaMa Model Loaded Once and Ready!")

    def predict(self, image_path, mask_path, output_path="/app/outputs/image_mask.png", refine=False):
        """Runs inference on a single image."""
        batch = self._prepare_input(image_path, mask_path)

        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch["mask"] = (batch["mask"] > 0).float().to(self.device)

            if refine:
                assert "unpad_to_size" in batch, "Unpadded size is required for refinement"
                cur_res = refine_predict(batch, self.model)
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                batch = self.model(batch)
                cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cur_res)
        return output_path

    def _prepare_input(self, image_path, mask_path):
        """Prepares input image and mask as tensors."""
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Error loading image or mask: {image_path}, {mask_path}")

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        batch = default_collate([{"image": image, "mask": mask}])
        return batch


if __name__ == "__main__":
    LOGGER.info("🚀 Starting LaMa Inpainting Service...")

    # ✅ Load Model ONCE
    model = LaMaModel()

    # ✅ Keep Listening for Requests
    for line in sys.stdin:
        try:
            image_path, mask_path = line.strip().split()
            LOGGER.info(f"Processing: {image_path}, {mask_path}")
            model.predict(image_path, mask_path)
            LOGGER.info("✅ Processing complete.")
        except Exception as ex:
            LOGGER.error(f"❌ Error: {ex}\n{traceback.format_exc()}")
