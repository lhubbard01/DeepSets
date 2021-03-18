"""inspired by PrototypicalNetworks/protonets/utils/model.py"""
from tqdm import tqdm

import torch
def evaluate(state,engine, opt, dataloader, meters, desc=None):
    model = state["model"]
    model.eval()

    for field,meter in meters.items():
        meter.reset()
    if desc is not None:
        dataloader = tqdm(dataloader, desc=desc)

    for (d,t) in dataloader:
      state["data"], state["targets"] = d, t
      engine.hooks["on_forward_pre"](state)
      engine.hooks["on_forward"](state)

      loss = state["criterion"](state["model_out"], 
                                  state["targets"]/state["targets"].size(0))
      fields = {"loss": loss.item(), "acc" : (1 if (loss**(1/2) < opt["train.acc"]) else 0 )}
      for field, meter in meters.items():
            meter.add(fields[field])

    model.train()
    return meters
