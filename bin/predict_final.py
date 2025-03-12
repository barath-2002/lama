import os
import logging
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

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

        self.model.to(self.device)  # No freeze()

        print("✅ Model loaded successfully!")

    def predict_image(self, image, mask):
        """Run the inpainting model on the given image and mask."""

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # Shape: (3, H, W)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # Shape: (1, H, W)

         # Add batch dimension
        batch = {
            'image': image_tensor.unsqueeze(0).to(self.device),  # Shape: (1, 3, H, W)
            'mask': (mask_tensor.unsqueeze(0).to(self.device) > 0).float(),  # Shape: (1, 1, H, W)
        }

        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = ((batch['mask'] > 0)).float().to(self.device)
            batch = self.model(batch)

            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

        return cur_res
