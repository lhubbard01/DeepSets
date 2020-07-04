import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class Conv(nn.Module):
  def __init__(self, phi1,phi2,phi3=None):
    super(Conv,self).__init__()
    self.phi1 = phi1
    self.phi2 = phi2
    self.phi3 = phi3
  def forward(self,X):
    X = self.phi1(X.reshape(-1,1,28,28))
    out=self.phi2(X.reshape(X.size(0),-1))
    out = self.phi3(out)
    return out





def train_conv(phi,savepath):
    phi.train()
    optimizer = optim.SGD(phi.parameters(),lr=0.001)
    crossEnt = nn.CrossEntropyLoss()
    accuracy = []
    losses = []

    for epoch in range(20):
      correct = 0
      for b, (d,t) in tqdm(train_convl,desc="epoch {epoch} of conv net training"):
        optimizer.zero_grad()
        out = phi(d).T
        
        loss = crossEnt(out.T.squeeze(), t)
        if b % 500 == 0:
          print(f"predictions: {out.cpu().argmax(dim=0)}\ntargets: {t}\naccuracy: {correct/((b+1)*32)}\nloss: {loss.item()}\n\n")
        acc_batch = (out.cpu().argmax(dim=0) == t.cpu()).float().sum()
        accuracy.append(acc_batch/32); correct+=acc_batch
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
      torch.save(phi,os.path.join(savepath+"__phi.pt"))

      print(f"Accuracy for epoch {epoch}: {(correct/len(train_convl.dataset))}")
