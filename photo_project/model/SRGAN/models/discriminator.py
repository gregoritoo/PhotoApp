import torch 
import torch.nn as nn 




class DiscriminatorBlock(nn.Module):

    def __init__(self,in_channel,out_channel,stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,(3,3),(stride,stride))
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()


    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        self.activation(x)
        return x 
    

class Discriminator(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channel,64,(3,3),(1,1))
        self.activation = nn.LeakyReLU()
        parameters= [(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2),(256,512,1),(512,512,2)]
        self.blocks = [DiscriminatorBlock(parameters[i][0],parameters[i][1],parameters[i][2]).cuda() for i in range(len(parameters))]
        self.dense_1 = nn.Linear(512*17*17*12,1024)
        self.activation_dense= nn.LeakyReLU()
        self.out =nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        x = self.input_conv(x)
        x = self.activation(x)
        for layer in self.blocks :
            x = layer(x)
        print(x.shape)
        x = self.dense_1(x.flatten())
        x = self.activation_dense(x)
        x = self.sigmoid(x)
        return x