import os
import logging
import torch
import numpy as np
import cv2
import tqdm
from pathlib import Path
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)

# Set environment variables for performance optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


class ImageInpainter:
    def __init__(self, predict_config):
        """
        Initializes the ImageInpainter by loading the model.

        :param predict_config: Configuration dictionary containing model details.
        """
        self.predict_config = predict_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = predict_config.model.path
        checkpoint = predict_config.model.checkpoint

        # ✅ Load training config
        train_config_path = os.path.join(model_path, 'config.yaml')
        if not os.path.exists(train_config_path):
            raise FileNotFoundError(f"Training config file not found at {train_config_path}")

        train_config = torch.load(train_config_path, map_location=self.device)
        train_config['training_model']['predict_only'] = True
        train_config['visualizer']['kind'] = 'noop'

        # ✅ Load model checkpoint
        checkpoint_path = os.path.join(model_path, checkpoint)
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)
        self.model.freeze()

        if not self.predict_config.get('refine', False):
            self.model.to(self.device)

    def predict_batch(self, indir: str, outdir: str, out_ext: str = ".png"):
        """
        Processes a batch of images from a directory.

        :param indir: Input directory containing images and masks.
        :param outdir: Output directory to save inpainted images.
        :param out_ext: File extension for output images.
        """
        try:
            indir = indir.rstrip('/') + '/'  # Ensure trailing slash
            dataset = make_default_val_dataset(indir, **self.predict_config.dataset)
            os.makedirs(outdir, exist_ok=True)

            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    outdir, os.path.splitext(mask_fname[len(indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = default_collate([dataset[img_i]])

                if self.predict_config.get('refine', False):
                    if 'unpad_to_size' not in batch:
                        raise ValueError("Refinement mode requires 'unpad_to_size' in batch.")
                    cur_res = refine_predict(batch, self.model, **self.predict_config.refiner)
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, self.device)
                        batch['mask'] = (batch['mask'] > 0).float().to(self.device)
                        batch = self.model(batch)
                        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)

            LOGGER.info("Processing complete.")

        except Exception as e:
            LOGGER.error(f"Prediction failed due to: {e}")
            raise RuntimeError(f"Prediction failed due to: {e}")
