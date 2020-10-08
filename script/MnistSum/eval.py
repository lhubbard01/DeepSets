import argparse
import deepsets.datasets.DataLoadWrapper as dlw
import torch
import random

















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
subset_size = input("enter the size of the subset to generate")

while str(subset_size) != "":
	data, target = dl.generate_new_subset(
		[random.randint(0,59999) for i in range(int(subset_size))], 
			dl.datamain, dl.targets)
	
	print(f"deepset mnist sum output    {  model(data.view(-1,1,784)).item()  }")
	print(f"target value    {target}")
	subset_size = input("enter the size of the subset to generate")
