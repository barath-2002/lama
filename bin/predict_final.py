import os
import logging
import numpy as np
import cv2
import torch
import yaml
import traceback
from pathlib import Path
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

# Disable threading optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Directories
OUTPUT_DIR = "/app/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

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

        # Ensure we use the correct output key from config.yaml
        if "evaluator" in train_config and hasattr(train_config.evaluator, "inpainted_key"):
            self.inpainted_key = train_config.evaluator.inpainted_key
        else:
            raise KeyError("❌ Missing `inpainted_key` in `train_config.evaluator`")

    def predict_image(self, image_path, mask_path):
        """Run the inpainting model, save results in `/app/outputs/`, and return output path."""

        try:
            # Load input image and mask
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                raise ValueError("❌ Failed to load input image or mask.")

            # Ensure image and mask are the same size
            H, W, _ = image.shape
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)  # Resize mask

            # Convert image & mask to float32
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

            # Ensure mask has 3 dimensions (H, W, 1)
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)

            batch = {
                'image': torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device),  # (1, 3, H, W)
                'mask': torch.from_numpy(mask).permute(2, 0, 1).float().unsqueeze(0).to(self.device),  # (1, 3, H, W)
            }

            with torch.no_grad():
                batch = move_to_device(batch, self.device)  # Move to GPU/CPU
                batch['mask'] = ((batch['mask'] > 0)).float().to(self.device)  # Ensure correct mask processing
                batch = self.model(batch)  # Run inference

                # Use the correct output key (from config.yaml, not predict_config.out_key)
                output_key = self.inpainted_key  
                cur_res = batch[output_key][0].permute(1, 2, 0).detach().cpu().numpy()

                # Handle unpadding if needed
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

            # Ensure correct scaling and format
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

            # Convert from RGB to BGR (OpenCV format)
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

            # Save output image
            output_path = os.path.join(OUTPUT_DIR, Path(image_path).stem + "_output.png")
            cv2.imwrite(output_path, cur_res)

            return output_path  # Return path instead of deleting file

        except Exception as ex:
            LOGGER.critical(f"Prediction failed due to {ex}:\n{traceback.format_exc()}")
            return {"error": f"Processing failed: {str(ex)}"}
