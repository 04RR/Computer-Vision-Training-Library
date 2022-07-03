from torchcam.methods import SmoothGradCAMpp
import cv2
import torch
from torchvision.transforms.functional import normalize


def get_activationMap(model, image, device='cpu'):

    cam_extractor = SmoothGradCAMpp(model)

    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float()
        image = normalize(image / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    out = model(image.unsqueeze(0).to(device))

    return cam_extractor(out.squeeze(0).argmax().item(), out)
