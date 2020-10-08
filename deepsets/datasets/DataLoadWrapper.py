import torch
import torch.nn as nn
import torchvision as tv
import copy
import random




def load_mnist_dataset(root:str, 
                       download : bool, 
                      train: bool = True):



  mnist_ds = tv.datasets.MNIST(root, 
                               train = train, 
                               download = download,
                               transform = tv.transforms.Compose(
                                  [tv.transforms.ToTensor()]
                                  )
                               )
  return mnist_ds

class DataLoadWrapper:

  """This is a standin to be passed to the engine via the state dict"""
  def __init__(self,
               data, 
               targets, 
               indices: list = None, 
               minset: int = 2, 
               maxset: int = 10):

    self.datamain = data
    self.targets  = targets
    self.minset   = minset
    self.maxset   = maxset

    if indices is not None: 
      self.indices = indices

    else:
      self.indices = self.gen_indices(self.datamain, self.datamain.size(0), 2, 10)
    
    self.length = len(self.indices)
  
  def __getitem__(self, index):
    data = self.datamain[ self.shuf(self.indices[index]) ]\
               .clone()\
               .detach()\
               .to(dtype=torch.float32) / 255


    targets = self.targets[self.indices[index]].sum() \
                    * torch.ones(data.size(0),
                                 dtype = torch.float32)

    return data, targets
  

  def shuffle_indices(self):
    """shuffle the actual subset locations"""
    indices = self.indices
    random.shuffle(indices)
    self.indices = indices
    return self.indices

  

  def shuf(self, index_list):
    """shuffle locations within a subset"""
    random.shuffle(index_list)
    return index_list
  
  def generate_new_subset(self, indices: list, data, targets):
    lis = [data[i] for i in indices]
    t   = sum([targets[i] for i in indices])

    out = torch.cat(lis).to(dtype = torch.float32) / 255 #normalize pixel intensities to between 0 and 1
    
    return out.view(len(indices),1,28,28), t


  def get_more(self, train_data, indices: list, min: int, max: int, count :int):
    for i in range(count):
      indices += self.gen_indices(train_data.shape[0],train_data,min,max)
    return indices

  def gen_indices(self, dataset,dataN:int, min:int, max_setsize:int):
    x = dataN

    lis = []
    while x > max_setsize:
      o = random.randint(min,max_setsize)
      lis.append(o)
      x-=o
    lis.append(x)
    
    lissubs = torch.utils.data.random_split(dataset,lis)
    print(f"{len(lissubs)} subsets generated\n")

    subsets_list = list(map(lambda x : x.indices, lissubs))
    return subsets_list
