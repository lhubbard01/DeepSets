import torchvision

import torchvision.transforms as transforms
MNIST = torchvision.datasets.MNIST

class DeepSetDatasetMnistTargetSum(MNIST):
  def __init__(self,
               root,
               train=True,
               transform=None,
               target_transform=None,
               download=False,
               max_set_size:int=9):
    
    MNIST.__init__( self, 
                    root,
                    train=True,
                    transform=transform,
                    target_transform=target_transform,
                    download=download)
    
    self.max_set_size = max_set_size




    
    

    x = self.data.shape[0]
    set_sizes = []
    while x > self.max_set_size:
      _size = np.random.randint(3,self.max_set_size)
      set_sizes.append(_size)
      x-=_size
    set_sizes.append(x)
    subsets = torch.utils.data.random_split(self.data,set_sizes) 
    self.subset_locations = []
    for i in range(len(subsets)):
      target = self.targets[subsets[i].indices].sum(dim=0,dtype=torch.float32,keepdims=True)
      self.subset_locations.append((subsets[i].indices,target))

    """if isinstance(dataset, torchvision.datasets.VisionDataset):
      self.transform = dataset.transform"""

  def __getitem__(self,index):
    """creates set items on the fly... maybe not feasible?"""
    #n_retrieved = index % self.max_set_size + 2 
    subset_tup = self.subset_locations[index]
    data, targets = self.data[subset_tup[0]], subset_tup[1]

    #indices = np.random.randint(0,len(self),n_retrieved)
    #data, targets = self.data[indices], self.targets[indices]
    out = []
    for i in range(data.shape[0]):

      img = data[i,:]
      img = Image.fromarray(img.numpy(), mode='L')


      if self.transform is not None:
          img = self.transform(img)
      if self.target_transform is not None:
          targets = self.target_transform(targets)  
      out.append(img)
    #print(f"out {out}, type, {type(out)}")
    out = torch.cat(out[:], 0)
    out = out.to(torch.float32)
    out /= 255


    #(out, device="cuda")
    return out, targets
  def __len__(self):
    return len(self.subset_locations)

def dataloader():
  pass
"""
dsDataset = DeepSetDatasetMnistTargetSum("./data/",
                             train=True,
                             transform=
                                transforms.Compose([transforms.ToTensor()]),
                             download=True,
                             max_set_size=9)"""