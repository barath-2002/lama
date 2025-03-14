from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import logging
import os
import shutil
import time
import torch
import cv2
import yaml
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Directories
INPUT_DIR = "/app/user_uploads/"
OUTPUT_DIR = "/app/outputs/"
MODEL_PATH = "/app/big-lama"
CHECKPOINT_PATH = f"{MODEL_PATH}/fine-tuned_lama.ckpt"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable for the model
loaded_model = None

def load_model():
    """Load the inpainting model only once when the app starts."""
    global loaded_model
    if loaded_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_config_path = os.path.join(MODEL_PATH, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        loaded_model = load_checkpoint(train_config, CHECKPOINT_PATH, strict=False, map_location=device)
        loaded_model.freeze()
        loaded_model.to(device)
        LOGGER.info("Model loaded successfully.")

# Initialize the model when the app starts
load_model()

app = FastAPI()

@app.post("/process/")
async def process_image(image: UploadFile = File(...), mask: UploadFile = File(...)):
    """Processes an image and mask using the preloaded model and returns the result."""
    global loaded_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save uploaded image & mask
    image_path = Path(INPUT_DIR) / "image.png"
    mask_path = Path(INPUT_DIR) / "image_mask.png"
    
    with image_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    with mask_path.open("wb") as buffer:
        shutil.copyfileobj(mask.file, buffer)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    dataset = make_default_val_dataset(INPUT_DIR)
    batch = move_to_device(default_collate([dataset[0]]), device)
    batch['mask'] = ((batch['mask'] > 0)).float().to(device)
    batch = loaded_model(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    output_image_path = Path(OUTPUT_DIR) / "output.png"
    cv2.imwrite(str(output_image_path), cur_res)
    
    # End timing
    processing_time = time.time() - start_time
    
    if output_image_path.exists():
        response = FileResponse(output_image_path, media_type="image/png")
        response.headers["X-Processing-Time"] = f"{processing_time:.2f} seconds"
        return response
    
    return {"error": "Inpainting failed", "processing_time": f"{processing_time:.2f} seconds"}
