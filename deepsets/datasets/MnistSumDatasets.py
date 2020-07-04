import torch
import torch.utils.data as torchdata
import torchvision
import numpy as np
from PIL import Image

__all__ = [
    "MnistSumDataSet",
    "MnistSumDataSet_Container"];




class MnistSumDataSet(torchdata.Dataset):
    """ A wrapper for the set of mnist raw images mapping to their sum """
    def __init__(self,
                 X,
                single_t,
                inherit_transforms=True,
                transform=None,
                target_transform=None,
                shuffle_tensor_form=True,


                ):
        
        self.X                   = X
        self.targets             = single_t
        self.transform           = transform
        self.target_transform    = target_transform
        self.shuffle_tensor_form = shuffle_tensor_form 


    def __getitem__(self,index):
        if self.shuffle_tensor_form:
            data, targets = self.return_shuffled_tensor()
        else:
            data, targets = self.X[index]
        img = data

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:

            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)


        return img, targets

    def return_shuffled_tensor(self):
        return self.X.random_(0,to=len(self))


    def __len__(self):
        return self.X.shape[0]

class MnistSumDataSet_Container(torchdata.Dataset):
    """MNIST data storage, subset storage, and producer of the data sets
    for the mnist sum tast"""
    def __init__(self,
                mnist_data=None, 
                max_set_size:int=9,
                download:bool=False,
                device="cpu",
                inherit_transforms=True,
                transform=None,
                target_transform=None):

        if mnist_data is None and download:
            torchvision.datasets.Mnist()#configure to load, etc
        elif mnist_data is None and not download:
            raise ValueError("no 'mnist_data' set and 'download' is False. mnist_data must either be passed in, or download must be set to True")
        else:
            raise ValueError("'mnist_data' must be set, or download must be set to True")
        
        assert isinstance(mnist_data, torchdata.Dataset) \
                or (isinstance(mnist_data, tuple) and isinstance(mnist_data[0], np.ndarray)),\
                "'mnist_data' must either be a Dataset or a tuple of numpy arrays"

        self.device             = device
        self.inherit_transforms = inherit_transforms 
        self.transform          = transform
        self.target_transform   = target_transform

        if isinstance(mnist_data, tuple):
            if isinstance(mnist_data[0], np.ndarray) and isinstance(mnist_data[1], np.ndarray):
                data_np,target_np = mnist_data[0], mnist_data[1]
                self.data     = torch.tensor(data_np, dtype=torch.uint8,   device=self.device)
                self.targets  = torch.tensor(target_np, dtype=torch.uint8, device=self.device)


        else:
            self.data  = mnist_data.data.clone().detach()
            self.targets = mnist_data.targets.clone().detach()

            if self.device == "cpu":
                self.data    = self.data.cpu()
                self.targets = self.targets.cpu() 

            elif self.device == "cuda":
                self.data    = self.data.cuda()
                self.targets = self.targets.cuda()

            else: raise ValueError(f"'device' must either be \"cpu\" or \"cuda\": device specified: {self.device}")

        if isinstance(mnist_data, torchvision.datasets.VisionDataset) and self.inherit_transforms:
            self.transform, self.target_transform = mnist_data.transform, mnist_data.target_transform
        self.sets = []
        self.max_set_size = max_set_size
        self.createSetMapping(self.data, self.targets)

    def generateSubsets(self, X, subsets:list):
        return torchdata.random_split(X, subsets)

    def createSetMapping(self, X, targets):

        x = X.shape[0] # size of all elements summed, needed for random_split
        subsets_lengths = []
        
        while x > self.max_set_size: #probably a better way to do this, was in a hurry
            set_size = np.random.randint(0,self.max_set_size)
            x -= set_size
            subsets_lengths.append(set_size)

        subsets_lengths.append(x)
        subsets = self.generateSubsets(X,subsets_lengths)

        for i in range(len(subsets)):

            target = targets[subsets[i].indices].sum()
            new_data = X[subsets[i].indices].clone().detach()

            sum_set = MnistSumDataSet(
                new_data,
                target * torch.ones((1)),
                transform = self.transform,
                target_transform = self.target_transform
                )
            
            sum_set_data_loader = torchdata.DataLoader(sum_set,batch_size=len(sum_set),shuffle=True)
            self.sets.append((sum_set, sum_set_data_loader))

