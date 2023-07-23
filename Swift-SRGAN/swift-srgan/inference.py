import os
import math
import pandas as pd
import torch
import torchvision
from data import TrainDataset, ValDataset, display_transform
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from loss import GeneratorLoss
from metric import ssim
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt 
import cv2
from torchvision import transforms
import numpy as np

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)


if __name__ == "__main__" :
    UPSCALE_FACTOR = 4
    DEVICE = torch.device("cpu")
    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    checkpoint = torch.load("./checkpoints/netG_4x_epoch100.pth.tar")
    netG.load_state_dict(checkpoint['model'])

    gt_image = cv2.imread("../dataset/test.png").astype(np.float32) / 255.
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
    print(gt_image.shape)
    gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/2),int(gt_image.shape[1]/4)), cv2.INTER_CUBIC)
    
    img = plt.imshow(gt_image)
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig('demo_real.png', bbox_inches='tight')
    gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0).to(DEVICE)



    sr_img = netG(gt_tensor)

    sr_image = transforms.ToPILImage()(sr_img.squeeze().detach().cpu())

    img = plt.imshow(sr_image)
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig('demo_upgrade.png', bbox_inches='tight')