"""influenced by Jake Snell et al Prototypical networks for few shot learning,
PrototypicalNetworks/scripts/train/few_shot/train.py"""


import os
import json
from functools import partial
from tqdm import tqdm

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchnet as tnt
from deepsets.engine import Engine


import deepsets
import deepsets.utils.model as model_utils
import deepsets.utils.log as log_utils

import deepsets.model.deepset as deepset
from deepsets.model.deepset import Phi, Rho
from trainRuntimeHelpers import *
from deepsets.datasets.DataLoadWrapper import DataLoadWrapper, load_mnist_dataset





"""main takes in parsed command line arguments, which have either been overriden 
   by specified assignments or tags, or are their default. main hooks callbacks to
   various defined hooks inside of the engine. """





separator = "*" * 80

def main(opt):
  opt["batch_size"] = opt["train.batch_size"]

  if not os.path.isdir(opt['log.experiment_directory']):
    os.makedirs(opt['log.experiment_directory'])

  with open(os.path.join(opt['log.experiment_directory'], 'options.json'),'w') as f:
    json.dump(opt,f)
    f.write("\n"+separator+"\n")


  trace = os.path.join(opt["log.experiment_directory"], "trace_file.txt")
  opt["log.fields"] = opt["log.fields"].split(",")

  def trainval(opt):
    #python import system is ass, look up mutual top level imports in python
    from deepsets import datasets as ds
    #if opt["data.train_validation"]:
    #  train_loader = DataLoadload_mnist_dataset("./mnist/",download=True,train=True)
    #  val_loader   = DataLoadWrapper(
    #else:
    data = load_mnist_dataset(root="./mnist/",train=True,download=True)
    train_loader = DataLoadWrapper(data.data,data.targets, minset=2,maxset=10 )
    return train_loader

  train_loader = trainval(opt)
  val_loader = None

  
  meters ={"train" : {field: tnt.meter.AverageValueMeter() for field in opt["log.fields"]}}
  if val_loader is not None:
    meters["validation"] = {field: tnt.meter.AverageValueMeter() for field in opt["log.fields"]}


  if opt["train.optimization_method"].lower() == "adam":  opt["optimization_method"] = torch.optim.Adam
  elif opt["train.optimization_method"].lower() == "sgd": opt["optimization_method"] = torch.optim.SGD



  engine = Engine()


  def on_start_with_visuals(state):
    return None

  def on_start(state):
    if opt["model.cuda"]:
      state["loader"].datamain.cuda(), state["loader"].targets.cuda()

    if os.path.isfile(trace):
      os.remove(trace)

  engine.hooks["on_start"] = on_start


  
  def on_start_epoch(state):
    for split, split_meters in meters.items():
      for field,meter in split_meters.items():
        meter.reset()
  engine.hooks["on_start_epoch"] = on_start_epoch



  


  def on_forward(state):
    out = state["model"](state["data"].view(-1,1,784))
    loss = state["criterion"](out, state["targets"])/state["targets"].size(0) #rescales gradient magnitude to match the singular output
    return loss, {"loss":loss.item(), 
                  "acc" : (1 if (loss**(1/2) < opt["train.acc"]) else 0 )}

  engine.hooks["on_forward"] = on_forward
 



  def on_update(state):
    for field, meter, in meters["train"].items():
      meter.add(state["output"][field])
  engine.hooks["on_update"] = on_update 





  def on_end_epoch(hook_state, state):
    if val_loader is not None:
      if "best_loss" not in hook_state:
        hook_state["best_loss"] = np.inf
      
      model_utils.evaluate(state,
                             val_loader,
                             meters["validation"],
                             desc="Epoch {:d} validation run ".format(state["epoch"]))

    meter_values = log_utils.extract_meter_values(meters)
    update_str = "Epoch {:02d}: {:s}".format(state["epoch"],
                              log_utils.render_meter_values(meter_values))
    
    print(update_str)

    meter_values["epoch"] = state["epoch"]
    state["loader"].shuf(state["loader"].indices)

    with open(trace, "a") as f:
      json.dump(meter_values,f)
      f.write("\n")


    if val_loader is not None:
      if meter_values["validation"]["loss"] < hook_state["best_loss"]:
            hook_state["best_loss"] = meter_values["validation"]["loss"]

            print("best model:  loss = {:0.6f}".format(hook_state["best_loss"]))

            state["model"].cpu()
            torch.save(state["model"], 
                       os.path.join(opt["log.experiment_directory"], "bestModel.pt"))
            
            if opt["data.cuda"]:
              state["model"].cuda()
      
      else:
          state["model"].cpu()
          torch.save(state["model"],
                     os.path.join(opt["log.experiment_directory"], "bestModel.pt"))
          
          if opt["data.cuda"]:
            state["model"].cuda()
    
    else:
          state["model"].cpu()
          torch.save(state["model"],
                     os.path.join(opt["log.experiment_directory"], str(state["epoch"]) + "model.pt"))

          if opt["data.cuda"]:
            state["model"].cuda()
    

    state["epoch"] += 1
  #End on_end_epoch 

  engine.hooks["on_end_epoch"] = partial(on_end_epoch, {})
  deepset_mnist_sum = load_model(opt)
  print("model loaded... running...")
  engine.train( 
          model=deepset_mnist_sum,
          epoch=opt["model.path.epoch"],
          loader=train_loader, 
          optimization_method=opt["optimization_method"],
          optim_config={"lr":opt["train.learning_rate"]},
          max_epochs=opt["train.epochs"],

          criterion=nn.MSELoss(reduction="mean"),
          name=opt["model.name"]
          )


