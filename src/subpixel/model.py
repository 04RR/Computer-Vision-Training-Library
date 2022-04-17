import torch
import torch.nn as nn
import json
from torchinfo import summary


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
                
                raise json.JSONDecodeError("layers not arranged properly!!","",-1)
            else:
                x = self.arch[i]["layer"]
                y = eval(x)
                self.architecture.append(y)
        summary(self,self.details["input_shape"])
    def forward(self,*X):
        outputs = []
        if len(X)!= self.details["num_inputs"]:
            l  = self.details["num_inputs"]
            raise RuntimeError(f"Expected {l} inputs, got {len(X)}.")
            return

        for i in range(len(self.arch)):
            if len(self.arch[i]["inputs"])==1:
                outputs.append(self.architecture[i](outputs[self.arch[i]["inputs"][0]] if self.arch[i]["inputs"][0]>=0 else X[abs(self.arch[i]["inputs"][0])-1]))
            elif len(self.arch[i]["inputs"])>=1:
                x = torch.cat([outputs[j] if j>=0 else X[abs(j)-1] for j in self.arch[i]["inputs"]], self.arch[i]["cat_dim"])
                outputs.append(self.architecture[i](x))
        
        return [outputs[j] for j in self.details["outputs"]] if len(self.details["outputs"])>1 else outputs[self.details["outputs"][0]] 



                
m = Model()
