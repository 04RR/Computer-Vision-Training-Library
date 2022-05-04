from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from model import Model
from PIL import Image

dataset = torchvision.datasets.FashionMNIST("./", download=True)


class Datas(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return (
            torch.tensor(np.array(self.dataset[index][0])).unsqueeze(0).float().cuda(),
            torch.tensor([1 if i == self.dataset[index][1] else 0 for i in range(10)]).float().cuda()
        )

    def __len__(self):
        # return 1000   
        return len(self.dataset)


class Test:
    def __init__(self, model, dataset, loss_fun) -> None:
        self.model = model
        # x = int(0.1*len(dataset)) if int(len(dataset))<100 else 100
        self.dataset= dataset
        self.loss_fun = loss_fun

    def test(self):
        print("Testing!")
        self.model.fit(self.dataset, self.loss_fun)


datase = Datas(dataset)
model = Model().cuda()
tes = Test(model, datase, loss_fun= nn.MSELoss())
tes.test()

