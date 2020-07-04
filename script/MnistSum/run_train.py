#!/bin/usr/python3
import argparse
from collections import namedtuple
import os
from tqdm import tqdm
from deepsets.engine import *
from train import main
















class Defaults:
  def __init__(self):
    self.data = {
        "path"                : "./mnist/",
        "dataset"             : "mnist",
        "minimum_subset_size" : 2,
        "maximum_subset_size" : 10,
        "train_validation"    : False,
        "cuda"                : False
    }
    self.model = {
        "name"                : "DeepSet",
        "path"                : None, #"./exp/",
        "path.epoch"          : 0,
        "cuda"                : False,
        "freeze_phi"          : False,
        "freeze_rho"          : False,
        "phi_activations"     : "relu",
        "rho_activations"     : "relu",
        "deepset_activations" : None, #note, setting this overrides phi and rho
    }
    self.train = {
        "epochs"              : 100,
        "visuals"             : False,
        "optimization_method" : "adam",
        "cuda"                : False,
        "learning_rate"       : 1e-4,
        "batch_size"          : 1,
        "acc"                 : 0.3,
    }
    self.log = {
        "fields"              : "loss,acc",
        "experiment_directory": "./exp/"
    }

defaults = Defaults()
parser = argparse.ArgumentParser(description="Train DeepSet on sum mapping from mnist")

dataD = defaults.data
parser.add_argument("--data.path",type=str,default=dataD["path"])
parser.add_argument("--data.dataset",type=str,default=dataD["dataset"])
parser.add_argument("--data.minimum_subset_size", type=int,default=dataD["minimum_subset_size"])
parser.add_argument("--data.maximum_subset_size",type=int,help="",default=dataD["maximum_subset_size"])
parser.add_argument("--data.train_validation",action="store_true")
parser.add_argument("--data.cuda",action="store_true")

#model
modelD = defaults.model
parser.add_argument("--model.name",type=str,default=modelD["name"])
parser.add_argument("--model.path",type=str,default=modelD["path"])
parser.add_argument("--model.path.epoch",type=int,default=modelD["path.epoch"],help="was too lazy to write a regex for getting most recent epoch to resume training, enter it here instead")
parser.add_argument("--model.cuda",action="store_true")
parser.add_argument("--model.freeze_phi",action="store_true")
parser.add_argument("--model.freeze_rho",action="store_true")
parser.add_argument("--model.rho_activations",type=str,default=modelD["rho_activations"])
parser.add_argument("--model.phi_activations",type=str,default=modelD["phi_activations"])
parser.add_argument("--model.deepset_activations",type=str,default=modelD["deepset_activations"])

#train
trainD = defaults.train
parser.add_argument("--train.epochs",type=int,default=trainD["epochs"],help=f"number of overall training epochs, default {trainD['epochs']}")
#parser.add_argument("--train.visuals",action="store_true")
parser.add_argument("--train.optimization_method",type=str,default=trainD["optimization_method"])
parser.add_argument("--train.device", action="store_true")
parser.add_argument("--train.learning_rate",type=float,default=trainD["learning_rate"])
parser.add_argument("--train.batch_size",type=int,default=trainD["batch_size"])
parser.add_argument("--train.acc", type=float, default=trainD["acc"], help="threshold at which to add a discrete unit to measure closeness in this regression (eg reg loss < 0.5, add one")
#log
logD = defaults.log
parser.add_argument("--log.fields",type=str,default=logD["fields"])
parser.add_argument("--log.experiment_directory",type=str,default=logD["experiment_directory"])

opt = vars(parser.parse_args())
main(opt)


