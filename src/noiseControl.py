import torch
import torch.nn as nn
import math
from config import eta_1, eta_T, p, T
'''
Timestamp in our ResShift is 0 - 14 (the scalar value)
'''

def resshift_schedule(T=T, eta1=eta_1, etaT=eta_T, p=p):
    betas = [ ((t-1)/(T-1))**p * (T-1) for t in range(1, T+1) ]
    b0 = math.exp((1/(2*(T-1))) * math.log(etaT/eta1))
    eta = [ eta1 * (b0 ** b) for b in betas ]
    return torch.tensor(eta)
