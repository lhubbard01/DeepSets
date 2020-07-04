import argparse
import deepsets.datasets.DataLoadWrapper as dlw
import torch
import random

















parser = argparse.ArgumentParser(description="Evaluate the most recent trained model on new data")
parser.add_argument("-d", type=int, default=10,help="load new set of specified size for regression")
parser.add_argument("--mnist", type=str, help="location of mnist dataset for loading")
parser.add_argument("--model.path", type=str, help="path to model")
options = vars(parser.parse_args())
f = open(options["model.path"], "rb")
model = torch.load(f)
f.close()


dmnist = dlw.load_mnist_dataset(options["mnist"], train=True, download=False)
dl = dlw.DataLoadWrapper(dmnist.data, dmnist.targets)
new_data = dl.gennew([random.randint(0,59999) for i in range(options["d"])], dl.datamain, dl.targets)
model.eval()
print(f"deepset mnist sum output    {model(new_data[0].view(-1,1,784)).item()}")
print(f"target value    {new_data[1]}")