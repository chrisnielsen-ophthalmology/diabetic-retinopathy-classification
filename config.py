
"""
Code adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy
Defines a bunch of global variables used for training
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 0
CHECKPOINT_FILE = "b4_10.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True
OVERWRITE_MODELS = True
