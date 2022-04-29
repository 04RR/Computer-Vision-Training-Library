import json
import numpy
from torch.utils.data import DataLoader
import torch
import torchvision
from numpy import diff
from tqdm import tqdm


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
        x = self.find_batch_size()
        print(x)
        Dataloader = iter(DataLoader(self.dataset, x,True))
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda epoch: epoch+dx)
        print("finding best LR!!")
        self.model.train()
        for i in tqdm(range(0,self.steps)):
            data,label = next(Dataloader)
            pred = self.model(data)
            loss = self.loss_fn(pred,label)
            self.loss.append(loss.detach().cpu().numpy())
            self.lr.append(self.start_lr+i*dx)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            scheduler.step()


        return self.lr[numpy.argmin(diff(self.loss)/dx)],self.loss,self.lr


    def find_batch_size(self):
        return 64 if len(self.dataset)//self.steps >64 else len(self.dataset)//self.steps