import torch 
import torch.nn as nn 
import torchvision
from model.SRGAN.models.generator import Generator
from model.SRGAN.models.discriminator import Discriminator
import torch.optim as optim
from model.SRGAN.loss.loss import GeneratorLoss
from model.SRGAN.data.ImageNetDataloaders import TrainDatasetFromFolder,ValDatasetFromFolder
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    imagenet_data = torchvision.datasets.ImageNet('../data','train')
    train_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size= batch_size,
                                          shuffle=True,
                                          num_workers=4)
    imagenet_data = torchvision.datasets.ImageNet('../data','test')
    test_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size= batch_size,
                                          shuffle=True,
                                          num_workers=4)
    return train_loader, test_loader
    

def train(epochs = 100):
    generator_criterion = GeneratorLoss()
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=1, upscale_factor=2)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=2)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    generator = Generator(3)
    discriminator = Discriminator(3)
    generator.train()
    discriminator.train()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    optimizerG = optim.Adam(generator.parameters())
    optimizerD = optim.Adam(discriminator.parameters())

    for epoch in range(epochs) :
        for data,target in train_loader :
            discriminator.zero_grad()
            if torch.cuda.is_available():
                target.cuda()
            if torch.cuda.is_available():
                data.cuda()
            fake_img = generator(data)
            real_img = target

            fake_out = discriminator(fake_img)
            real_out = discriminator(real_img)
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            generator.zero_grad()

            fake_img = generator(data)
            fake_out = discriminator(fake_img)
            g_loss =  generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            print(f" Epoch : {epoch} generator loss : {g_loss} discriminator loss {d_loss}")





