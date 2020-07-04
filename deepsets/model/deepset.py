import torch
import torch.nn as nn


class DeepSet(nn.Module):
  def __init__(self, phi, rho):
    super(DeepSet, self).__init__()
    self.phi = phi
    self.rho = rho
  def forward(self,X):
    phi_out = self.phi(X)
    summation = phi_out.sum(dim=0,keepdim=True) 
    summation.requires_grad_(True)
    rho_out = self.rho(summation)
    return rho_out




class Phi(nn.Module):
  def __init__(self, lis:nn.ModuleList):
    super(Phi,self).__init__()
    self.lis = lis # meant to be nn.ModuleList
  def forward(self,X):
    for module in self.lis:
      X = module(X)
    return X

class Rho(nn.Module):
  def __init__(self,lis:nn.ModuleList):
    super(Rho, self).__init__()
    self.lis = lis

  def forward(self,X):
    for module in self.lis:
      X = module(X)
    return X
