import os
import math
import pandas as pd
import torch
import torchvision
from model.SRGAN.models.generator import Generator
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt 
import cv2
from torchvision import transforms
import numpy as np

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)
UPSCALE_FACTOR = 4

if __name__ == "__main__" :
    DEVICE = torch.device("cuda:0")
    netG = Generator().to(DEVICE)
    checkpoint = torch.load("model/SRGAN/checkpoints/netD_4x_epoch101.pth.tar")
    netG.load_state_dict(checkpoint['model'])
    path_to_image = "model/SRGAN/data/test.png" # Change with the path to your image
    gt_image = cv2.imread(path_to_image).astype(np.float32) / 255.
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
   
    ## To make it run in my small laptop, I first downscall the image
    ## but the following line is not mandatory 
    gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/2),int(gt_image.shape[1]/4)), cv2.INTER_CUBIC)
    img = plt.imshow(gt_image)
    # let's remove useless axes 
    plt.axis('off')
    plt.savefig('demo_real.png', bbox_inches='tight')
    print("gt_image",gt_image.shape)
    gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0).to(DEVICE)
    sr_img = netG(gt_tensor)
    print("sr_img",sr_img.shape)
    sr_image = transforms.ToPILImage()(sr_img.squeeze().detach().cpu())

    img = plt.imshow(sr_image)
    plt.axis('off')
    plt.savefig('demo_upgrade.png', bbox_inches='tight')