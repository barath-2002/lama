#!/usr/bin/env python3

# Example command:
# !python3 "/content/lama/bin/lama.py" \
# model.path="/content/drive/MyDrive/Magic Eraser/big-lama" \
# indir="/content/input" \
# outdir="/content/outputs" \
# model.checkpoint="/content/drive/MyDrive/Magic Eraser/lama/big-lama/last.ckpt"

import logging
import os
import sys
import traceback
import torch
import cv2
import hydra
import numpy as np
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

# Global variable to store the model
loaded_model = None

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    global loaded_model  # Use the global model variable

    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model only if it has not been loaded yet
        if loaded_model is None:
            train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))

            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            out_ext = predict_config.get('out_ext', '.png')

            checkpoint_path = os.path.join(predict_config.model.path, 
                                           'models', 
                                           predict_config.model.checkpoint)
            loaded_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
            loaded_model.freeze()
            loaded_model.to(device)
            LOGGER.info("Model loaded successfully.")
        else:
            LOGGER.info("Using preloaded model.")

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = ((batch['mask']>0)).float().to(device)
                batch = loaded_model(batch)  # Use the preloaded model
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
