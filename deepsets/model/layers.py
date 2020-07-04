import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = ["PermutationInvariance__func", "PermutationInvariance"]

class PermutationInvariance__func(Function):
	"""This function is used to perform"""
	@staticmethod
	def forward(ctx, X):
		summation = torch.ones_like(X) * X.clone().detach().sum(dim = 0, keepdims = True)
		ctx.save_for_backward(X, summation)
		return summation



	@staticmethod
	def backward(ctx, gradient):
		X, summation = ctx.saved_tensors
		summation_expanded = (torch.ones(
								size=(X.shape[0], X.shape[1])
							) * summation.clone().detach()).requires_grad_(True)

		grad_sum = summation_expanded.mm(gradient.T)
		grad_weight = X.mm(grad_sum)
		return grad_weight

class PermutationInvariance(nn.Module):
	"""This layer is used to induce a 'Permutation Invariance', ie
	properties analogous to mathematical sets. There is no 
	ordinal behavior, and this layer induces that property"""
	def __init__(self): 
		super(PermutationInvariance, self).__init__()

	def forward(self, X):
		return PermutationInvariance__func.apply(X)

