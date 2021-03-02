import os
import json
import torch
import torch.nn as nn
from deepsets.model import deepset














def gen_arch_phi(opt):
  """Options for setting activations from command line, as well as a vanilla architecture"""
  act = opt["model.phi_activations"].lower()
  if act == "relu":      act = nn.ReLU
  elif act == "tanh":    act = nn.tanh
  elif act == "sigmoid": act = nn.Sigmoid
  elif act == "erelu":   act = nn.EReLU
  elif act == "lrelu":   act = nn.LeakyReLU
  else: raise ValueError(f"unexpected activation for phi {act}")
  
  phi=deepset.Phi(nn.ModuleList(
      [nn.Linear(784,300),act(), 
      nn.Linear(300,100), act(), 
      nn.Linear(100,10)]))
  return phi

def gen_arch_rho(opt):
  """Options for setting activations from command line, as well as a vanilla architecture"""
  act = opt["model.rho_activations"].lower()
  if act == "relu":
    act = nn.ReLU
  elif act == "tanh":
    act = nn.tanh
  elif act == "sigmoid":
    act = nn.Sigmoid
  elif act == "erelu":
    act = nn.EReLU
  elif act == "lrelu":
    act = nn.LeakyReLU
  else:
    raise ValueError(f"unexpected activation for rho {act}")

  return deepset.Rho(nn.ModuleList([nn.Linear(10,1)]))



def hyperparam(opt,model,name):
  if opt["model.freeze_"+name]:
    model.eval()
  else:
    model.train()

  if opt["model.cuda"]:
    model.cuda()
  else:
    model.cpu()

  return model #note, model flags set via methods dont require capture. 
  #still returns self so is useful for method chaining

def load_model(opt):
  if not opt["model.set_phi"] and not opt["model.set_rho"] and opt["model.path"]:
    f = open(opt["model.path"],"rb")
    model = torch.load(f)
    f.close()
    return model

  rho = phi = None

  if opt["model.set_phi"]:
    f = open(opt["model.set_phi"],"rb")
    phi = torch.load(f)
    f.close()
  else:
      phi = gen_arch_phi(opt)

  if opt["model.set_rho"] :
    f = open(opt["model.set_rho"], "rb")
    rho = torch.load(f)
    f.close()
  else:
      rho = gen_arch_rho(opt)

  model = deepset.DeepSet(phi,rho)

  if opt["model.cuda"]:
    model.cuda()
  else:
    model.cpu()
  model.train()
  return model




