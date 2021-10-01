import torch
import torch.nn as nn


def inverse_huber_loss(output,target):
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
    

