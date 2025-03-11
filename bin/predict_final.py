#!/usr/bin/env python3

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

class LaMaPredictor:
    """Loads LaMa inpainting model once and allows multiple predictions."""

    def __init__(self, model_path):
        """Initialize model and load checkpoint."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"✅ Model will run on: {self.device}")

        # Load model configuration
        train_config_path = os.path.join(model_path, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        # Load model checkpoint
        checkpoint_path = os.path.join(model_path, "models", "fine-tuned_lama.ckpt")
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)
        self.model.freeze()
        self.model.to(self.device)

    def predict(self, input_dir, output_dir):
        """Run inference on input images using the preloaded model."""
        dataset = make_default_val_dataset(input_dir)
        results = []

        for img_i in tqdm.trange(len(dataset)):
            start_time = time.time()

            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                output_dir, os.path.basename(mask_fname)
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            batch = default_collate([dataset[img_i]])

            with torch.no_grad():
                batch = move_to_device(batch, self.device)
                batch['mask'] = ((batch['mask'] > 0)).float().to(self.device)
                batch = self.model(batch)

                cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get("unpad_to_size", None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

            end_time = time.time()
            processing_time = end_time - start_time
            results.append((mask_fname, processing_time))

        return results

# ✅ Load model once (to be used by FastAPI)
MODEL_PATH = "/app/big-lama"
lama_predictor = LaMaPredictor(MODEL_PATH)
