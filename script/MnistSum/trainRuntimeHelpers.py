import os
import json
import torch
import torch.nn as nn
from deepsets.model import deepset















def gen_arch_phi(opt):
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
  if opt["train.device"] is False: 
    phi.cpu() 
  else:
    phi.cuda()
  return phi

def gen_arch_rho(opt):
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

  return deepset.Rho(nn.ModuleList(nn.Linear(10,1)))



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
  if not opt["model.path"]:
    phi = gen_arch_phi(opt)
    rho = gen_arch_rho(opt)
    model = deepset.DeepSet(phi,rho)
    #model = deepset.DeepSet(phi,rho)
    if opt["model.cuda"]:
      model.cuda()
    else:
      model.cpu()

  else:
    f = open(opt["model.path"],"rb")
    model = torch.load(f)
    f.close()

  model.train()
  return model





def parse_json_model(path:str, module_paths:list, list_out:bool=False):
  """Takes the path as a location for where the model to be loaded resides.
    module_paths are searchable locations from which to instantiate the
    nn.Module the corresponding key from the current list position 
    (each list position is itself a dictionary, containing the name and either
    a list of positional arguments or a dictionary of assignable arguments)

    optional: list_out causes return of list of instantiated from the parse

    returns:
      nn.Sequential container of parsed objects
    """

  """function scoped helper function definitions"""
  def handle_dict(optionals:dict,  layer:str,  paths:list):
    """handles keyword and optional arguments"""
    layer_low = layer.lower()
    for path in paths:
      modules = path.__dir__()
      modules_low = list(map(str.lower, modules))
      if layer_low in modules_low:
        module = getattr(path, modules[modules_low.index(layer_low)])
        if "positionals" in optionals.keys():
          positionals =optionals["positionals"]
          optionals.pop("positionals")
          execstr="out = module("
          for i in range(len(positionals)):
            if i == 0:
              execstr+=str(positionals[i])
            else:
              execstr+=", "+str(positionals[i])
        else:
          execstr = "out = module("
        for k,v in optionals.items():
            execstr += ", " + str(k)+" = "+str(v)
            print(execstr)
        execstr += ")"
        print(execstr)        
        exec(execstr, globals()+locals())

        return out

  #def recur
  def handle_list(positionals:list, layer:str, paths:list):
    layer_low = layer.lower()

    for path in paths:
      modules_low = list(map(str.lower, path.__dir__()))
      if layer_low in modules_low:

            out = (getattr(path, path.__dir__()[modules_low.index(layer_low)]),
              positionals)
      else:
          raise ValueError("module {0} specified does not exist in the paths {1}".format(layer,paths))
      return out






  """parse_json_model main"""
  if not os.path.isfile(path):
    raise ValueError(f"Path to loadable json file containing model is either incorrect or doesn't exist\nfile path : {path}")
  with open(path,"r") as f:
    dictmodel= json.load(f)
  
  model=[]
  parsed = []

  for el in dictmodel["model"]:
    if isinstance(el, dict):
      keys = [k for k in el.keys()]
      for k in keys:
        if isinstance(el[k], dict):
          parsed.append(handle_dict(el[k], k, module_paths), 1)
        elif isinstance(el[k],list):
          parsed.append(handle_list(el[k],k,module_paths))
  
  for layer in parsed:
    layerClass, hyper = layer
    if isinstance(hyper,int):
      model.append(hyper)
    else:
      model.append(layerClass(*hyper))
  if list_out:
    return model
  else:
    return nn.Sequential(*model)
