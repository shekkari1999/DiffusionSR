import torch
import torch.nn as nn
from model import FullUNET
from noiseControl import resshift_schedule
from torch.utils.data import DataLoader
from data import train_dataset
import torch.optim as optim
from config import batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

hr, lr = next(iter(train_dl))
upsample_layer = nn.Upsample(scale_factor=4, mode = 'nearest')
lr_upsampled = upsample_layer(lr)
hr = hr.to(device)
lr_upsampled = lr_upsampled.to(device)
k = 1
eta = resshift_schedule().to(device)
eta = eta[:, None, None, None]   # shape (15,1,1,1)
residual = (lr_upsampled - hr)
model = FullUNET()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
steps = 150
for step in range(steps):
    model.train()
    # take random timestep
    t = torch.randint(0, 14, (batch_size,)).to(device)
    #print(t.dtype, t.min().item(), t.max().item())

    # add the noise
    epsilon = torch.randn_like(hr)
    eta_t = eta[t]
    x_t = hr + eta_t * residual + k * torch.sqrt(eta_t) * epsilon
    # send the same patch in model forwardpass across different timestamps per each step
    pred = model(x_t, t)
    optimizer.zero_grad()
    loss = criterion(pred, epsilon)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f'loss at step {step + 1} is {loss}')
