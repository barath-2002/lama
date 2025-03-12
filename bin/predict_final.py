import os
import logging
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

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

    def predict_image(self, image, mask):
        """Run the inpainting model on the given image and mask, treating both the same way."""

        # Ensure image and mask are the same size
        H, W, _ = image.shape
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)  # Ensure same dimensions

        # Convert image & mask to float32 for consistency
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Convert image & mask to PyTorch tensors (ensuring correct format)
        batch = {
            'image': torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device),  # (1, 3, H, W)
            'mask': torch.from_numpy(mask).permute(2, 0, 1).float().unsqueeze(0).to(self.device),  # (1, 3, H, W)
        }

        with torch.no_grad():
            batch = move_to_device(batch, self.device)  # Move to GPU/CPU
            batch = self.model(batch)  # Run inference

            cur_res = batch['inpainted'][0]  # Extract inpainted result

            # If the result is a PyTorch tensor, convert it back to NumPy
            if isinstance(cur_res, torch.Tensor):
                cur_res = cur_res.permute(1, 2, 0).detach().cpu().numpy()

            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        # Ensure correct scaling and format
        cur_res = (cur_res * 255).astype(np.uint8)
        cur_res = np.clip(cur_res, 0, 255)
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)  # Convert for OpenCV compatibility

        return cur_res
