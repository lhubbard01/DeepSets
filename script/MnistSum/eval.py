import argparse
import deepsets.datasets.DataLoadWrapper as dlw
import torch
import random
import torchvision as tv
import matplotlib.pyplot as plt
















parser = argparse.ArgumentParser(description="Evaluate the most recent trained model on new data")

parser.add_argument("--mnist", type=str, help="location of mnist dataset for loading")
parser.add_argument("--model.path", type=str, help="path to model")

options = vars(parser.parse_args())

f = open(options["model.path"], "rb")
model = torch.load(f)
f.close()


dmnist = dlw.load_mnist_dataset(options["mnist"], train=True, download=False)
dl = dlw.DataLoadWrapper(dmnist.data, dmnist.targets)

model.eval()

def percent_err(target, model_output):
	return (abs(target - model_output) / target) * 100

def asgrid(tensor, rows = 8):
	grid = tv.utils.make_grid(tensor, nrow = rows, pad_value = 0.5)
	print(f"shape of grid from tensor {tensor.shape} is:  {grid.shape}")
	try:
		plt.imshow(grid.numpy().transpose(1,2,0))
	except TypeError:
		plt.imshow(grid.numpy())
	plt.show()


def add_num_to_subset(tensor, dl, number: int, prev_target: int):
	if tensor is not None: 
		print(f"input tensor shape: {tensor.shape}")
	all_as_lbl = dl.datamain[dl.targets == number]
	x = all_as_lbl.to(dtype = torch.float32)[random.randint(0,len(all_as_lbl))]
	if tensor is None:
		return x.reshape(1,1,28,28) / 255.0, number
	else:
		return torch.cat((tensor, x.reshape(1,1,28,28) / 255.0), dim = 0), number + prev_target


def cumulative_subsets():
	""" allows the same subset to persist across trials, while also allowing for additional images to contribute"""
	added_num = int(input("first number?\t"))
	data, target = add_num_to_subset(None, dl, added_num, 0)	
	disp = input("display values chosen?\t")
	
	while str(added_num) != "":
		output = model(data.view(-1,1,28,28)).item()
		print(f"deepset mnist sum output    {  output  }")
		print(f"target value    {target}")
		print(f"percent error is {percent_err(target, output)} %")
		if disp:
			asgrid(data)
		print("\n\n")



		if "y" in input("add many?\t"):

			added_num = int(input("the number\t"))
			lim = int(input("how many times\t"))

			for i in range(lim):
				data, target = add_num_to_subset(data, dl, added_num, target)

		else:
			added_num = input("add number between 0 and 9\t")
			data, target = add_num_to_subset(data, dl, int(added_num), target)

def subset_generation_by_size():
	""" Specify a subset size. These will likely be novel data for the model"""
	subset_size = input("enter the size of the subset to generate\t")
	while str(subset_size) != "":
		data, target = dl.generate_new_subset(
			[random.randint(0,59999) for i in range(int(subset_size))], 
				dl.datamain, dl.targets)
	
		disp = (True if "y" in input("display values chosen?\t") else False)
		output = model(data.view(-1,1,28,28)).item()
		print(f"deepset mnist sum output    {output}")
		print(f"target value    {target}")
		print(f"percent error is {percent_err(target, output)} %")
		if disp:
			asgrid(data)
		subset_size = input("enter the size of the subset to generate\t")


as_cumulative_subsets = (True if "y" in input("per iteration additional images? y/n\t") else False)
if as_cumulative_subsets:
	cumulative_subsets()
else:
	subset_generation_by_size()