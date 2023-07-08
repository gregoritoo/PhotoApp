import torch 
import numpy as np



if __name__ == '__main__' :
    y = torch.zeros((1,4))
    print(y.to("cuda:0"))