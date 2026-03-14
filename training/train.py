import torch

def train(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        loss = batch.mean()
        loss.backward()
        optimizer.step()