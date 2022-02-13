import json
import torch


def show_batch(data):
    pass


def EncodingToClass(lst, classes):

    lst = list(lst.detach().squeeze(0).numpy())
    return classes[lst.index(max(lst))]


def get_boxxes(t):
    # '{x, y, h, w, [classes]}' -> [x, y, h, w, classes]
    bbox = list(json.loads(t).values())
    return bbox[:-1] + bbox[-1]

def accuracy(out, labels):
    
    c=0
    
    preds = torch.round(out)
    preds = preds.detach().cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    
    for label, pred in zip(labels, preds):
        if pred == label:
            c+=1

    return c/len(out)
