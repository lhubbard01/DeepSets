import torch
import torch.nn as nn

import copy
import random









class DataLoadWrapper:
  """This is a standin to be passed to the engine via the state dict, 
  yielding the subsets via iter calls."""
  def __init__(self, dataMain, targets, indices:list=None, minset:int=2, maxset:int=10):
    self.datamain = dataMain
    self.targets = targets

    self.minset = minset
    self.maxset = maxset
    if indices is not None: 
      self.indices = indices
    else:
      self.indices = self.gen_indices(self.datamain, self.datamain.size(0), 2, 10)

  def __getitem__(self,index):
    data = self.shuf(self.datamain[self.indices[index]]),
    targets = self.targets[self.indices[index]].sum() * torch.ones(d.shape[0],dtype=torch.float32)
    return data, targets
    
  def shuf(self, index_list):
    indices1 = copy.deepcopy(index_list) #index list is treated as a "ground truth" list
    random.shuffle(indices1)
  return indices1
  
  def gennew(self, indices:list,dataset,labels):
    lis = [dataset[i] for i in indices]
    t = sum([labels[i] for i in indices])
    out = torch.cat(lis).to(dtype=torch.float32)/255 #normalize pixel intensities to between 0 and 1
    return out.view(len(indices),1,28,28), t

  def getmore(self, train_data, indices:list, min:int, max:int,count:int):
    for i in range(count):
      indices += self.gen_indices(train_data.shape[0],train_data,min,max)
    return indices

  def gen_indices(self, dataN:int,dataset,min:int, max_setsize:int):
    x = dataN
    lis = []
    while x > max_setsize:
      o = random.randint(min,max_setsize)
      lis.append(o)
      x-=o
    lis.append(x)
    lissubs = torch.utils.data.random_split(dataset,lis)
    return lissubs