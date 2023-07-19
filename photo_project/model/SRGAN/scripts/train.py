import torch 
import torch.nn as nn 
import torchvision
from model.SRGAN.models.generator import Generator
from model.SRGAN.models.discriminator import Discriminator
import torch.optim as optim
from model.SRGAN.loss.loss import GeneratorLoss
from model.SRGAN.data.ImageNetDataloaders import TrainDatasetFromFolder,ValDatasetFromFolder
from torch.utils.data import DataLoader

    

def train(epochs = 100):
    generator_criterion = GeneratorLoss()
    train_set = TrainDatasetFromFolder('model/SRGAN/data/VOC_train_HR', crop_size=186, upscale_factor=2)
    val_set = ValDatasetFromFolder('model/SRGAN/data/VOC_test_HR', upscale_factor=2)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
    generator = Generator(3)
    discriminator = Discriminator(3)
    generator.train()
    discriminator.train()
    if torch.cuda.is_available():
        generator = generator.to("cuda:0")
        discriminator = discriminator.to("cuda:0")
    optimizerG = optim.Adam(generator.parameters())
    optimizerD = optim.Adam(discriminator.parameters())
    print("generator ",next(generator.parameters()).is_cuda)
    for epoch in range(epochs) :
        for data,target in train_loader :
            discriminator.zero_grad()
            if torch.cuda.is_available():
                target = target.to("cuda:0")
            if torch.cuda.is_available():
                data = data.to("cuda:0")
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



if __name__ == '__main__' :
    train(1)

