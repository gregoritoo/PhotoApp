import torch.nn as nn
import torch 



class ResidualBlock(nn.Module) :
    def __init__(self,input_size=64):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_size,64,(3,3),(1,1))
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64,64,(3,3),(1,1))
        self.batch_norm_2 = nn.BatchNorm2d(64)


    def forward(self,x) :
        self.residue = x 
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = nn.PReLU()(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = torch.add(x,self.residue)
        return x
    

class ShufllingBlock(nn.Module):
    def __init__(self,channel,upscale_factor=2):
        super().__init__()
        self.conv_layer = nn.Conv2d(channel,256,(3,3),(1,1))
        self.pixel_shufller = nn.PixelShuffle(upscale_factor)
        self.activation = nn.PReLU()


    def forward(self,x):
        x = self.conv_layer(x)
        x = self.pixel_shufller(x)
        x = self.activation(x)
        return x 


    


class Generator(nn.Module):
    """ 
    Generator Encoder from https://arxiv.org/pdf/1609.04802.pdf
    
    """
    def __init__(self,input_channel=3,nb_resiudal_blocks=5):
        self.conv_layer = nn.Conv2d(input_channel,64,9,1)
        self.norm = nn.BatchNorm2d(64)
        self.residual_blocks =[ResidualBlock(64) for _ in range(nb_resiudal_blocks)] 
        self.in_conv_layer = nn.Conv2d(64,3,3,1)
        self.norm_2 = nn.BatchNorm2d(64)
        self.out_conv_layer = nn.Conv2d(256,2,(3,3),(1,1))
        self.upscaling_layers = [ShufllingBlock(64,2) for _ in range(2) ]


    def forward(self,x):
        x = self.conv_layer(x)
        x = self.norm(x)
        self.residue_1 = x 
        for layer in self.residual_blocks :
            x = layer(x)
        x = self.in_conv_layer(x)
        x = self.norm_2(x)
        x = torch.add(x,self.residue_1)
        for layer in self.upscaling_layers :
            x = layer(x)
        return x 


