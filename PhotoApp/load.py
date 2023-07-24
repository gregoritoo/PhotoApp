import os
import math
import pandas as pd
import torch
import torchvision
from models.model import Generator
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt 
import cv2
from torchvision import transforms
import numpy as np

torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(42)


UPSCALE_FACTOR = 4
DEVICE = torch.device("cuda:0")

def load_trained_model(path= "./models/weights/netD_4x_epoch101.pth.tar") :
    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['model'])
    return netG


