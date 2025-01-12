import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from itertools import zip_longest
from torch.utils.data import DataLoader
from types import SimpleNamespace

from model.patchtst import Model
from dataset import Stock

DEVICE="cpu"
BATCH_SIZE=128

EPOCHS=50
WINDOW=32
LR=1e-4
PCT_START = 0.3

args = {
    "seq_len":WINDOW,
    "pred_len":1,
    "enc_in":6,
    "e_layers":3,
    "n_heads":16,
    "d_model":32,  # must be divisible by nheads
    "d_ff":256,
    "dropout":0.3,
    "fc_dropout":0.3,
    "head_dropout":0,
    "patch_len":16,
    "stride":8,
    "individual":1,
    "padding_patch":"end",
    "revin":1,
    "affine":1,
    "subtract_last":0,
    "decomposition":1,
    "kernel_size":25,
    "pe":"sincos",
    "learn_pe":False
    }

args = SimpleNamespace(**args)

stocks = ["btc-usd"] 

trainset = Stock(stocks=stocks, window=WINDOW, train=True, test_year=2024)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

model = Model(configs=args).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
                                    steps_per_epoch = len(trainloader),
                                    pct_start = PCT_START,
                                    epochs = EPOCHS,
                                    max_lr = LR)

for epoch in range(EPOCHS):    
    epoch_loss=0
    for inputs, targets in tqdm(trainloader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        pred = model(inputs)
        pred = pred.squeeze(1)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss+=loss.item()
    print(f'for epoch {epoch+1} ; loss is {epoch_loss}')


checkpoint = {
    'model':model.state_dict(),
    'args':args
    }
torch.save(checkpoint, f"checkpoints/model.pkl")