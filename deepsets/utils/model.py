"""Copied directly from Jake Snell et al Prototypical networks for few shot learning, 
PrototypicalNetworks/protonets/utils/model.py"""
from tqdm import tqdm
import torch
def evaluate(state, dataloader, meters, desc=None):
    model = state["model"]
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        dataloader = tqdm(dataloader, desc=desc)

    for (d,t) in dataloader:
        output=state["model"](d)
        acc_val = torch.eq(torch.argmax(output), t).float().mean()
        loss = state["criterion"](output,t)


        fields = {"loss": loss.item(), "acc" : acc_val.item()}
        for field, meter in meters.items():
            meter.add(fields[field])

    model.train()
    return meters
