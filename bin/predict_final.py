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
        with open(config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Load model checkpoint
        checkpoint_path = os.path.join(model_path, checkpoint_name)
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)
        self.model.freeze().to(self.device)
    
    def predict_image(self, image, mask):
        """Run the inpainting model on the given image and mask."""

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float()

        # Add batch dimension
        batch = {
            'image': image_tensor.unsqueeze(0).to(self.device),
            'mask': (mask_tensor.unsqueeze(0).to(self.device) > 0).float(),
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
