import os
import logging
import torch
import yaml
import tqdm
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import logging
import os
import sys
import traceback

# Add the logging configuration here
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOGGER = logging.getLogger(__name__)

# Your existing imports and code below...
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

# Your main function...

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


class ImageInpainter:
    def __init__(self, predict_config: dict):
        """
        Initializes the ImageInpainter by loading the model.

        :param predict_config: Dictionary containing model and dataset configuration.
        """
        # ✅ Convert dictionary to OmegaConf object (to maintain dot notation)
        self.predict_config = OmegaConf.create(predict_config)

        # ✅ Correct way to access values now
        model_path = self.predict_config.model.path
        checkpoint = self.predict_config.model.checkpoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load training config
        train_config_path = os.path.join(model_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # Load the model checkpoint
        checkpoint_path = os.path.join(model_path, 'models', checkpoint)
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
            if not indir.endswith('/'):
                indir += '/'
    
            # Load dataset using the config
            dataset = make_default_val_dataset(indir, **self.predict_config.dataset)
            os.makedirs(outdir, exist_ok=True)
    
            for img_i in tqdm.trange(len(dataset)):
                # Ensure we're accessing the correct mask filename
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    outdir, os.path.splitext(mask_fname[len(indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
    
                # Collect the batch from the dataset
                batch = default_collate([dataset[img_i]])
    
                # Handle refinement if enabled
                if self.predict_config.get('refine', False):
                    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                    cur_res = refine_predict(batch, self.model, **self.predict_config.refiner)
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, self.device)
                        batch['mask'] = ((batch['mask'] > 0)).float().to(self.device)
                        batch = self.model(batch)
                        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]
    
                # Save the inpainted result
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
    
            LOGGER.info("Processing complete.")
        except Exception as e:
            LOGGER.error(f"Error processing batch: {e}")
            raise e
