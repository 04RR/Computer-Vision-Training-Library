from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import json
from torchinfo import summary
from train import Trainer
from utils import FindLR
import numpy as np


class Model(nn.Module):
    def __init__(self,path = 'arch.json') -> None:
        super(Model,self).__init__()

        JSON_file = open(path,"r")
        arch = json.load(JSON_file)
        JSON_file.close()
        
        self.details = arch["details"]
        self.arch = arch["arch"]
        self.architecture = nn.ModuleList()

        for i in range(len(self.arch)):
            if self.arch[i]["code"] != i:
                
                raise json.JSONDecodeError("layers not arranged properly!!", "", -1)
            else:
                x = self.arch[i]["layer"]
                y = eval(x)
                self.architecture.append(y)
        
        #summary(self,self.details["input_shape"])
    
    def forward(self,*X):

        outputs = []
        
        if len(X)!= self.details["num_inputs"]:

            l  = self.details["num_inputs"]
            raise RuntimeError(f"Expected {l} inputs, got {len(X)}.")
            
            return

        for i in range(len(self.arch)):

            if len(self.arch[i]["inputs"]) == 1:
                outputs.append(self.architecture[i](outputs[self.arch[i]["inputs"][0]] if self.arch[i]["inputs"][0] >= 0 else X[abs(self.arch[i]["inputs"][0]) - 1]))
            
            elif len(self.arch[i]["inputs"])>=1:
                x = torch.cat([outputs[j] if j>=0 else X[abs(j)-1] for j in self.arch[i]["inputs"]], self.arch[i]["cat_dim"])
                outputs.append(self.architecture[i](x))
        
        return [outputs[j] for j in self.details["outputs"]] if len(self.details["outputs"]) > 1 else outputs[self.details["outputs"][0]] 

    def fit(self,trainset):

        self.idealLR, self.loss,self.LRs = FindLR(self, trainset, nn.MSELoss(), torch.optim.Adam(self.parameters())).findLR()
        plt.plot(self.LRs,self.loss)
        plt.show()
        # self.trainer = Trainer(self,trainset,None,1,"classification",learning_rate=self.idealLR)
        # self.history = self.trainer.fit()

    def find_size(self):
    # added new funcion
        p_total = sum(p.numel() for p in self.parameters() if p.requires_grad) 
        bits = 32.

        mods = list(self.modules())
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            sizes = []
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        total_bits = 0
        for i in range(len(sizes)):
            s = sizes[i]
            bits = np.prod(np.array(s))*bits
            total_bits += bits

        return p_total, total_bits

                
m = Model()
