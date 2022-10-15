import torch

PRED_TH = 0.1
NMS_TH = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
