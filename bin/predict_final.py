import os
import logging
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate

# Disable threading optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

LOGGER = logging.getLogger(__name__)

class LaMaModel:
    def __init__(self, model_path, checkpoint_name="fine-tuned_lama.ckpt"):
        """Load the model once at startup."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Load model configuration
        config_path = os.path.join(model_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file missing: {config_path}")

        checkpoint_path = os.path.join(model_path, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint missing: {checkpoint_path}")

        print(f"🔍 Loading model from: {checkpoint_path} on {self.device}")

        with open(config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Load model checkpoint
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)
        if self.model is None:
            raise RuntimeError("❌ Model failed to load!")

        self.model.to(self.device)  # Move model to GPU/CPU
        print("✅ Model loaded successfully!")

        # Ensure the correct output key is used
        if "evaluator" in train_config and hasattr(train_config.evaluator, "inpainted_key"):
            self.out_key = train_config.evaluator.inpainted_key
        else:
            self.out_key = "inpainted"  # Default key

    def predict_image(self, image, mask):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))  # (H, W, C) → (C, H, W)
        sample = {'image': image, 'mask': mask}

        batch = default_collate([sample])  # Now batch will be (B, C, H, W)
    
        with torch.no_grad():
            batch = move_to_device(batch, self.device)  # Move to GPU/CPU
            batch['mask'] = (batch['mask'] > 0).float().to(self.device)  # Ensure correct mask processing
            batch = self.model(batch)  # Run inference
    
            # Ensure output key exists
            if self.out_key not in batch:
                raise KeyError(f"❌ Expected output key '{self.out_key}' not found in model output.")
    
            # Retrieve the correct output tensor
            cur_res = batch[self.out_key][0].permute(1, 2, 0).detach().cpu().numpy()  # (C, H, W) → (H, W, C)

        cur_res = np.clip(cur_res, 0, 255).astype(np.uint8)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    
        return cur_res
