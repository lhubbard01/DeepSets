import tqdm

class Engine:
  """A training time management abstraction. Adapted from torchnet
  and prototypical networks for few-shot learning, implemented by jake snell
  
  hooks are callbacks that are used at train. a single dictionary 'state'
  is passed to each callback. If no callback is hooked to a hook, 
  will return 'None'.

  
  Hook names =     "on_start", "on_start_epoch",
                  "on_forward_pre", "on_forward",
                  "on_backward",
                  "on_end_epoch", "on_end"
  



  




  Note: User is expected to define :
    1) optimizer behavior in the callbacks, including zeroing, stepping, etc
    2) calling backward on the loss 
  """
  
  def __init__(self, notebook: bool = False):
    """suggested hooks to attach"""
    hook_names = ["on_start", "on_start_epoch",
                  "on_forward_pre", "on_forward",
                  "on_backward",
                  "on_end_epoch", "on_end", "on_update"]
    self.hooks = {k : lambda state: None for k in hook_names } 
    self.as_notebook = notebook
    if notebook:
      self.tqdmcb = tqdm.autonotebook.tqdm
    else:
      self.tqdmcb = tqdm.tqdm
    self.train = self.train_reg
    """default to None so if used in training loop without having been hooked,will still run"""
  

  def train(self,**kwargs):
    state = {
        "model"         : kwargs["model"],
        "loader"        : kwargs["loader"],
        "optim_method"  : kwargs["optimization_method"],
        "optim_config"  : kwargs["optim_config"],
        "max_epoch"     : kwargs["max_epochs"],
        "name"          : kwargs["name"],
        "criterion"     : kwargs["criterion"],
        "epoch"         : kwargs["epoch"],
        "t"             : 0,
        "accuracy"      : 0,
        "stop"          : False,
        "optimizer"     : None,
        "output"        : None,
        "data"          : None, 
        "targets"       : None #only for reshaping in forward_pre hook
        }

    state["optimizer"] = state["optim_method"](
        state["model"].parameters(), 
        **state["optim_config"]
        )
    self.hooks["on_start"](state)
    while state["epoch"] < state["max_epoch"] and not state["stop"]:
      self.hooks["on_start_epoch"](state)
      for i,(d,t) in self.tqdmcb(enumerate(state["loader"]), desc="epoch {:d} training".format(state["epoch"])): #retrieve subset structures
            state["data"], state["targets"] = d, t
            self.hooks["on_forward_pre"](state)
            self.hooks["on_forward"](state)
            self.hooks["on_backward"](state)
            self.hooks["on_update"](state)
      self.hooks["on_end_epoch"](state)
    self.hooks["on_end"](state)

