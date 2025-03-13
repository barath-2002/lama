import logging
import os
import torch
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
import cv2
import tqdm
from omegaconf import OmegaConf
import yaml

LOGGER = logging.getLogger(__name__)

class ModelInitializer:
    def __init__(self, model_path: str, checkpoint: str, config_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the model by loading the checkpoint."""
        try:
            train_config_path = os.path.join(self.model_path, 'config.yaml')
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            checkpoint_path = os.path.join(self.model_path, 'models', self.checkpoint)
            self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=self.device)

            self.model.freeze()
            self.model.to(self.device)
            LOGGER.info("Model initialized successfully.")
        except Exception as e:
            LOGGER.error(f"Error initializing model: {e}")
            raise

    def predict(self, predict_config: OmegaConf, image_data: dict):
        """Performs image processing and returns the result."""
        try:
            if not predict_config.indir.endswith('/'):
                predict_config.indir += '/'
            
            # Prepare the dataset
            dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
            mask_fname = dataset.mask_filenames[0]  # Assuming processing one image per call
            cur_out_fname = os.path.join(predict_config.outdir, os.path.splitext(mask_fname[len(predict_config.indir):])[0] + predict_config.get('out_ext', '.png'))

            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[0]])
            batch = move_to_device(batch, self.device)
            batch['mask'] = ((batch['mask'] > 0)).float().to(self.device)

            with torch.no_grad():
                batch = self.model(batch)
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
            
            return cur_out_fname
        except Exception as e:
            LOGGER.error(f"Prediction failed: {e}")
            return None
