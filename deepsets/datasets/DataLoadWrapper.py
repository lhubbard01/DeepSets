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
  """This is a standin to be passed to the engine via the state dict.
   Target_callback allows for defining own label generation at runtime"""
  def __init__(self,
               data, 
               targets, 
               indices: list = None, 
               minset: int = 2, 
               maxset: int = 10,
               num_subsets: int = 0,
               target_callback = None):

    self.datamain = data
    self.targets  = targets

    self.minset   = minset
    self.maxset   = maxset

    if target_callback is None:
      self.target_callback = label_sum
    else:
      self.target_callback = target_callback

    if indices is not None: 
      self.indices = indices
    elif num_subsets != 0:
      self.indices = self.gen_indices_disjoint(self.datamain, self.datamain.size(0), 2, 10)
    else:
      self.indices = self.gen_indices_with_replacement(self.datamain, self.datamain.size(0), 2, 10, num_subsets)
    self.length = len(self.indices)
  
  def __getitem__(self, index):
    """Used when object is treated as a generator. Fetches the subset at index, 
      whose contents are shuffled and used to retrieve the image data from the data matrix.
      The corresponding target labels are summed and subsequently mapped with a ones vector.
      This can
      
      NOTE: This vector is used during training, the subset's cardinality DIVIDES THE TARGET VECTOR
      when generating the loss signal.  This has to be done in pytorch (as far as I know) in order 
      to skirt some autograd rules. The issue of regression when considering the images not as a 
      singular trainable entity, but insetad as a batch, causes this.

      returns - data tensor and target vector
      """

    data = self.datamain[ self.shuf(self.indices[index]) ]\
               .clone()\
               .detach()\
               .to(dtype=torch.float32) / 255 # treat data as their own memory, normalize


    targets = self.target_callback(self, index, data)
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


  def generate_additional_subsets(self, train_data, indices: list, min: int, max: int, count :int):
    for i in range(count):
      indices += self.gen_indices(train_data.shape[0], train_data, min, max)
    return indices

  def gen_indices_with_replacement(self, dataset, dataset_size: int, min: int, max: int, , number_of_subsets: int):
    """ This method allows for the generation of a specified number of subsets, created from with-replacement sampling 
      @dataset_size - upper bounds the random sampling
      @number_of_subsets - the number of subsets to be generated through with replacement datapoint sampling  
      @min - smallest subset size
      @max - largest subset size

      returns - list of lists containing indices, each list constituting a subset
    """
    indices = []
    for i in range(number_of_subsets):
      subset_size = random.randint(min, max)
      indices.append([random.randint(0,dataset_size) for j in range(subset_size)])
    
    return indices 

  def gen_indices_disjoint(self, dataset, dataset_size: int, min: int, max: int):
    """ This method allows for the generation of a specified number of subsets, created from with-replacement sampling 
      @dataset - used in partitioning the data
      @dataset_size - is used in generating subset sizes
      @min - smallest subset size
      @max - largest subset size

      returns - list of lists containing indices, each list constituting a subset
      """
    x = dataset_size

    disjoint_subset_size_list = []
    # while there remain data to be partitioned into disjoint subsets, select between the min and max size
    while x > max:
      subset_size = random.randint(min, max)
      disjoint_subset_size_list.append(subset_size)
      x -= subset_size
    disjoint_subset_size_list.append(x)
    
    
    # Note, since this is leveraging the random_split method, the data are partitioned, i.e. are disjoint.
    # Typically this method is used to create train, validate, and test splits. However, it works nicely here
    # lissubs is an intermediate value of sorts
    # subsets_list contains lists of indices making up the subsets. That is, elements of each subset are indices 
    # mapping to images, which are yielded from the dataloader at runtime.  
    lissubs = torch.utils.data.random_split(dataset, disjoint_subset_size_list)
    print(f"{len(lissubs)} subsets generated\n")

    subsets_list = list(map(lambda x : x.indices, lissubs))
    return subsets_list

def label_sum(obj, index, data):
  return obj.targets[obj.indices[index]].sum() \
                    * torch.ones(data.size(0),
                                 dtype = torch.float32)



import functools
def label_mult(obj, index, data):
  out = int(functools.reduce(lambda x, y: x * y, obj.targets[obj.indices[index]]))
  return out * torch.ones(data.size(0), dtype = torch.float32)
