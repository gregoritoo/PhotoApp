import torch 
import torch.nn as nn 
import torchvision
from model.SRGAN.models.generator import Generator
from model.SRGAN.models.discriminator import Discriminator
import torch.optim as optim
from model.SRGAN.loss.loss import GeneratorLoss
from model.SRGAN.data.ImageNetDataloaders import TrainDatasetFromFolder,ValDatasetFromFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np 

DEVICE = "cuda:0"   
UP_FACTOR = 4

def train_model(epochs = 100):
    generator_criterion = GeneratorLoss(DEVICE)
    train_set = TrainDatasetFromFolder('model/SRGAN/data/train', crop_size=96, upscale_factor=UP_FACTOR)
    val_set = ValDatasetFromFolder('model/SRGAN/data/valid', upscale_factor=UP_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
    generator = Generator(3)
    discriminator = Discriminator(3)
    if torch.cuda.is_available():
        generator = generator.to(DEVICE)
        discriminator = discriminator.to(DEVICE)
    optimizerG = optim.Adam(generator.parameters(), lr=1e-3)
    optimizerD = optim.Adam(discriminator.parameters(), lr=1e-3)
    generator.train()
    discriminator.train()
    path_to_image = "model/SRGAN/data/test.png" # Change with the path to your image
    gt_image = cv2.imread(path_to_image).astype(np.float32) / 255.
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
   
    ## To make it run in my small laptop, I first downscall the image
    ## but the following line is not mandatory 
    gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/2),int(gt_image.shape[1]/4)), cv2.INTER_CUBIC)
    gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0).to(DEVICE)
    
    print("Starting training")
    for epoch in range(epochs) :
        for data,target in tqdm(train_loader) :
            if torch.cuda.is_available():
                target = target.to(DEVICE)
                data = data.to(DEVICE)
            fake_img = generator(data)
            real_img = target

            discriminator.zero_grad()
            fake_out = discriminator(fake_img)
            real_out = discriminator(real_img)
            d_loss = 1 - real_out.mean() + fake_out.mean()
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            generator.zero_grad()

            fake_img = generator(data)
            fake_out = discriminator(fake_img).mean()

            g_loss =  generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            optimizerG.step()
            batch_mse = ((fake_img - real_img) ** 2).data.mean()

        print(f" Epoch : {epoch} generator loss : {g_loss} discriminator loss {d_loss} MSE {batch_mse}")
        sr_image = transforms.ToPILImage()(fake_img[-1,:,:,:].squeeze().detach().cpu())
        plt.imshow(sr_image)
        plt.axis('off')
        plt.savefig(f'fake_img_upgrade_epoch{epoch}.png', bbox_inches='tight')
        torch.save(
            {"model": generator.state_dict()},
            f"model/SRGAN/checkpoints/netD_{UP_FACTOR}x_epoch{epoch}.pth.tar",
        )
    generator.eval()

    with torch.no_grad():
        for batch,(data,_ ,target) in enumerate(val_loader) :
            discriminator.zero_grad()
            if torch.cuda.is_available():
                target = target.to(DEVICE)
                data = data.to(DEVICE)
            fake_img = generator(data)
            real_img = target

            batch_mse = ((fake_img - real_img) ** 2).data.mean()

        print(f"MSE values at batch {batch} is {batch_mse}")
    
if __name__ == '__main__' :
    train_model(102)

