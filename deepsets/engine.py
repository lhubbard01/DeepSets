from tqdm import tqdm
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
  """
  
  def __init__(self):
    """suggested hooks to attach"""
    hook_names = ["on_start", "on_start_epoch",
                  "on_forward_pre", "on_forward",
                  "on_backward",
                  "on_end_epoch", "on_end"]
    self.hooks = {k : lambda state: None for k in hook_names } 
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

      for i,(d,t) in tqdm(enumerate(state["loader"]), desc="epoch {:d} training".format(state["epoch"])): #retrieve subset structures
            state["data"], state["targets"] = d, t
            state["optimizer"].zero_grad()

            self.hooks["on_forward_pre"](state)

            loss, state["output"] = self.hooks["on_forward"](state)
            loss.backward()

            self.hooks["on_backward"](state)
            state["optimizer"].step()
            self.hooks["on_update"](state)
      self.hooks["on_end_epoch"](state)
    self.hooks["on_end"](state)




"""def on_start(state):
def on_start_epoch(state)
def on_forward_pre(state)
def on_forward(state)
def on_backward(state)
def on_end_epoch(state) 
def on_end(state)
class State(dict):
  def __init__(self,list):
    super(State, self).__init__()
    [self[key]  for key in list]
    self["model"]
    self["t"]
    self["max_epoch"]
    self["criterion"]
    self["loader"]
    self["optimizer"]"""


