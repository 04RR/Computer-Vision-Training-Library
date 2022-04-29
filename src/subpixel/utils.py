import json
import numpy
from torch.utils.data import DataLoader
import torch
import torchvision
from numpy import diff


def show_batch(data):
    pass


def EncodingToClass(lst, classes):

    lst = list(lst.detach().squeeze(0).numpy())
    return classes[lst.index(max(lst))]


def get_boxxes(t):
    # '{x, y, h, w, [classes]}' -> [x, y, h, w, classes]
    bbox = list(json.loads(t).values())
    return bbox[:-1] + bbox[-1]


class FindLR():
    def __init__(self,model, dataset, loss_fn,optimizer , start_lr = 1e-6, end_lr = 1e-3,steps = 20) -> None:
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.steps = steps
    def findLR(self):
        self.lr = []
        self.loss = []
        dx = (self.end_lr - self.start_lr)/self.steps
        Dataloader = DataLoader(self.dataset, self.find_batch_size(),True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda epoch: epoch+dx)
        self.model.train()
        for i in self.steps:
            data,label = next(iter(Dataloader[i]))
            pred = self.model(data)
            loss = self.loss(pred,label)
            self.loss.append(loss.detatch())
            self.lr.append(self.start_lr+i*self.steps)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            scheduler.step()


        return self.lr[numpy.argmin(diff(self.loss)/dx)],self.loss,self.lr


    def find_batch_size(self):
        #Needs to be changed.
        return 2