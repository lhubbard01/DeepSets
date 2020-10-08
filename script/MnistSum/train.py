"""influenced by Jake Snell et al Prototypical networks for few shot learning,
PrototypicalNetworks/scripts/train/few_shot/train.py"""

import os
import pickle 
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

  dataset = load_mnist_dataset(root="./mnist/",train=True,download=True)

  


  def train_load(opt):
    if opt["train.reuse"] is True:
      with open(os.path.join(opt["log.experiment_directory"], "training_set"), "rb") as f:
        train_loader = pickle.load(f)

    elif not opt["train.tv_split"]:
      train_loader = DataLoadWrapper(dataset.data,
                                     dataset.targets, 
                                     minset = 2,
                                     maxset = 10 )
      with open(os.path.join(opt["log.experiment_directory"], "training_set"), "wb") as f:
        pickle.dump(train_loader,f)

    else: 
      return None

    return train_loader
  

  


  def valid_load(opt):
    if opt["train.reuse"]:
      try:
        with open(os.path.join(opt["log.experiment_directory"], "valid_set"), "rb") as f:
          valid_loader = pickle.load(f)
      except Exception as e:
        raise e
    return valid_loader



  def gen_tv_split(opt):
    trs = torch.utils.data.random_split(dataset, [48000,12000])
    print(trs)
    only_indices = lambda X: X.indices
    train_split, valid_split = list(map(only_indices, trs))

    train_loader =  DataLoadWrapper(dataset.data[train_split], 
                                    dataset.targets[train_split],
                                    minset = 2,
                                    maxset = 10)
    valid_loader = DataLoadWrapper(dataset.data[valid_split], 
                                   dataset.targets[valid_split],
                                   minset = 2,
                                   maxset = 10)
    
    with open(os.path.join(opt["log.experiment_directory"], "training_set"), "wb") as f:
      pickle.dump(train_loader,f)
    
    with open(os.path.join(opt["log.experiment_directory"], "valid_set"), "wb") as f:
      pickle.dump(valid_loader,f)

    return train_loader, valid_loader

  train_loader = train_load(opt)
  valid_loader = None

  if opt["train.tv_split"] and opt["train.reuse"]:
    valid_loader = valid_load(opt)
  
  if opt["train.tv_split"] and not opt["train.reuse"]:
    train_loader, valid_loader = gen_tv_split(opt) 





  meters ={"train" : {field: tnt.meter.AverageValueMeter() for field in opt["log.fields"]}}
  if valid_loader is not None:
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
    if valid_loader is not None:
      if "best_loss" not in hook_state:
        hook_state["best_loss"] = np.inf
      
      model_utils.evaluate(state,
                             valid_loader,
                             meters["validation"],
                             desc="Epoch {:d} validation run ".format(state["epoch"]))

    meter_values = log_utils.extract_meter_values(meters)
    update_str = "Epoch {:02d}: {:s}".format(state["epoch"],
                              log_utils.render_meter_values(meter_values))
    
    print(update_str)

    meter_values["epoch"] = state["epoch"]
    
    state["loader"].shuffle_indices()

    with open(trace, "a") as f:
      json.dump(meter_values,f)
      f.write("\n")


    if valid_loader is not None:
      if meter_values["validation"]["loss"] < hook_state["best_loss"]:
            hook_state["best_loss"] = meter_values["validation"]["loss"]

            print("best model:  loss = {:0.6f}".format(hook_state["best_loss"]))

            state["model"].cpu()
            torch.save(state["model"], 
                       os.path.join(opt["log.experiment_directory"],str(state["epoch"]) + "__" + opt["model.name"] + ".pt"))
            if opt["model.cuda"]:
              state["model"].cuda()
      else:
          state["model"].cpu()
          torch.save(state["model"],
                     os.path.join(opt["log.experiment_directory"], str(state["epoch"]) + "__" + opt["model.name"] + ".pt"))
          
          if opt["data.cuda"]:
            state["model"].cuda()
    else:
          state["model"].cpu()
          torch.save(state["model"],
                     os.path.join(opt["log.experiment_directory"], str(state["epoch"]) + "__" + opt["model.name"]+".pt"))
          if opt["model.cuda"]:
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


