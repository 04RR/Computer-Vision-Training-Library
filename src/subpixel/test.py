from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from model import Model
from torchinfo import summary
from PIL import Image

dataset = torchvision.datasets.MNIST('./',download=True)
class Datas(torch.utils.data.Dataset):
    def __init__(self,dataset) -> None:
        super().__init__()
        self.dataset = dataset
    def __getitem__(self,index):
        return torch.tensor(np.array(self.dataset[index][0])).unsqueeze(0).float().cuda(),torch.tensor([1 if i==self.dataset[index][1] else 0 for i in range(10)]).float().cuda()
    def __len__(self):
        return int(0.1*len(dataset)) if int(0.1*len(dataset))<100 else 100



class Test:
    def __init__(self,model,dataset,loss_fn,optimizer) -> None:
        self.model = model
        #x = int(0.1*len(dataset)) if int(len(dataset))<100 else 100
        self.dataset, self.loss_fn, self.optimizer = dataset, loss_fn, optimizer
    
    def test(self):
        print("Testing......")
        self.model.fit(self.dataset)
        print("sorted!!!")
datase = Datas(dataset)
model = Model().cuda()
tes = Test(model,datase,nn.MSELoss(),torch.optim.Adam(model.parameters()))
tes.test()

