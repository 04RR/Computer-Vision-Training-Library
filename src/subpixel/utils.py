import json
import numpy
from torch.utils.data import DataLoader
import torch
import torchvision
from numpy import diff
from tqdm import tqdm
import torch.nn as nn
import os
import numpy as np
import random


def show_batch(data):
    pass


def EncodingToClass(lst, classes):

    lst = list(lst.detach().squeeze(0).numpy())
    return classes[lst.index(max(lst))]


def get_boxxes(t):
    # '{x, y, h, w, [classes]}' -> [x, y, h, w, classes]
    bbox = list(json.loads(t).values())
    return bbox[:-1] + bbox[-1]


def seed_everything(seed=42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def init_model(m):
    # print(list(m.modules()))
    # return [nn.init.xavier_uniform_(i.weight) for i in m.modules()]

    seed_everything()

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        # if m.bias is not None:
        #     nn.init.xavier_uniform_(m.bias.data)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)


class FindLR:
    def __init__(
        self, model, dataset, loss_fn, optimizer, start_lr=1e-6, end_lr=1e-3, steps=20
    ) -> None:

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

        dx = (self.end_lr - self.start_lr) / self.steps
        x = self.find_batch_size()

        print(x)

        Dataloader = iter(DataLoader(self.dataset, x, True))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: epoch + dx
        )

        self.model.train()

        self.model = self.model.cuda()
        self.model.apply(init_model)

        for i in tqdm(range(0, self.steps)):

            data, label = next(Dataloader)
            pred = self.model(data)
            loss = self.loss_fn(pred, label)

            self.loss.append(loss.detach().cpu().numpy())
            self.lr.append(self.start_lr + i * dx)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            scheduler.step()

        return self.lr[numpy.argmin(diff(self.loss) / dx)], self.loss, self.lr

    def find_batch_size(self):

        p, total_bits = self.model.find_size()
        f_before = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)

        for data in self.dataset:
            img, label = data
            img = img.cuda()
            label = label.cuda()
            break

        f_after = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        data_size = -f_after + f_before

        available_size = 0.95 * (f_after - total_bits + data_size)

        torch.cuda.empty_cache()
        b_size = int(available_size // data_size)

        return b_size if len(self.dataset) > b_size else len(self.dataset)
