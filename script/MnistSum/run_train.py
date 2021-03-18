#!/usr/bin/env python3
import argparse
from train import main



class Defaults:
  def __init__(self):
    self.data = {
        "path"                : "./mnist/",
        "dataset"             : "mnist",
        "minimum_subset_size" : 2,
        "maximum_subset_size" : 10,
        "num_subsets": 0,
    }
    self.model = {
        "name"                : "DeepSet",
        "set_phi"             : None,
        "set_rho"             : None,
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
        "reuse"               : False,
        "optimization_method" : "adam",
        "cuda"                : False,
        "learning_rate"       : 1e-4,
        "batch_size"          : 1,
        "acc"                 : 0.3,
        "tv_split"            : False,
        "accuracy_metric"     : None,
        "notebook"            : False,
    }
    self.log = {
        "fields"              : "loss,acc",
        "experiment_directory": "./exp/"
    }







defaults = Defaults()
parser = argparse.ArgumentParser(description="Train DeepSet on sum mapping from mnist")

dataD = defaults.data
parser.add_argument("--data.num_subsets",type=str,default=dataD["num_subsets"],
                    help=f"generates this many subsets, which is used in lieu of entirely disjoint, nonoverlapping subsets. 0 here means generate disjoint subsets. no upper bound on this. default={dataD['num_subsets']}")
parser.add_argument("--data.path",type=str,default=dataD["path"],
                    help=f"the path from which to read data at train time, default={dataD['path']}")

parser.add_argument("--data.dataset",type=str,default=dataD["dataset"],
                    help=f"dataset that is to be loaded (will be an added feature in the future) default={dataD['dataset']}")

parser.add_argument("--data.minimum_subset_size", type=int,default=dataD["minimum_subset_size"],
                    help=f"the minimum subset size to be used during permutation invariance training default={dataD['minimum_subset_size']}")

parser.add_argument("--data.maximum_subset_size",type=int,default=dataD["maximum_subset_size"],
                    help=f"the maxmimum subset size used during permutation invariance training, default={dataD['maximum_subset_size']}")

#model
modelD = defaults.model
parser.add_argument("--model.name",type=str,default=modelD["name"],
                    help=f"name to give the model being trained, will be used during saving as a suffix to the epoch number being saved. default={modelD['name']}")

parser.add_argument("--model.path",type=str,default=modelD["path"],
                    help=f"path from which to load or save model. default={modelD['path']}")

parser.add_argument("--model.path.epoch",type=int,default=modelD["path.epoch"],
                    help=f"will soon use regex to do this, in meantime, specify corresponding model to load from epoch to resume training, default={modelD['path.epoch']}")

parser.add_argument("--model.set_rho", type=str,default=None,
                    help=f"load a model from this path to plug into the deepset as the rho network, default={modelD['set_rho']}")

parser.add_argument("--model.set_phi", type=str,default=None,
                    help=f"load a model from this path to plug into the deepset as the phi network, default={modelD['set_phi']}")

parser.add_argument("--model.cuda",action="store_true", 
                    help="Set this flag to train model using cuda capable device, default is False")

parser.add_argument("--model.freeze_phi",action="store_true",
                    help="freeze phi weights during training, default is True")

parser.add_argument("--model.freeze_rho",action="store_true",
                    help="freeze rho weights during training, default is True")

parser.add_argument("--model.rho_activations",type=str,default=modelD["rho_activations"],
                    help=f"specify which nonlinearity to pass through products in rho network during training, default={modelD['rho_activations']}")

parser.add_argument("--model.phi_activations",type=str,default=modelD["phi_activations"],
                    help=f"specify which nonlinearity to pass through products in phi network during training, default={modelD['phi_activations']}")

parser.add_argument("--model.deepset_activations",type=str,default=modelD["deepset_activations"],
                    help=f"specify which nonlinearity to pass through products in entire network during training, default={modelD['deepset_activations']}\nNote: overrides other specified activations")

#train
trainD = defaults.train
parser.add_argument("--train.epochs",type=int,default=trainD["epochs"],help=f"number of overall training epochs, default {trainD['epochs']}")

parser.add_argument("--train.reuse", action="store_true", help="Set this flag to reuse previously generated subsets. Default is False" )

parser.add_argument("--train.optimization_method",type=str,default=trainD["optimization_method"],
                    help=f"optimiztation method to use during training. will be upgraded to use all available in torch. currently available are adam and sgd. default={trainD['optimization_method']}")

parser.add_argument("--train.learning_rate",type=float,default=trainD["learning_rate"],
                    help=f"learning rate to use during training, default={trainD['learning_rate']}")

parser.add_argument("--train.batch_size",type=int,default=trainD["batch_size"])

parser.add_argument("--train.acc", type=float, default=trainD["acc"], 
                    help="threshold at which to add a discrete unit to measure closeness in this regression (eg reg loss < 0.5, add one)")

parser.add_argument("--train.tv_split",action="store_true", help="train and validation split of 20 valid, 80 train")

parser.add_argument("--train.accuracy_metric", type=str, default = trainD["accuracy_metric"],
        help=f"a file containing a function of the same name, to calculate accuracy of model's predictions, default={trainD['accuracy_metric']}")

parser.add_argument("--train.notebook", action = "store_true",
        help=f"if this model is being launched from a terminal instance in notebook, flag sets it to true default={trainD['notebook']}")

#log
logD = defaults.log
parser.add_argument("--log.fields",type=str,default=logD["fields"],
                    help=f"which fields to save during training, currently in progress feature. defaults={logD['fields']}")

parser.add_argument("--log.experiment_directory",type=str,default=logD["experiment_directory"],
                    help=f"where to store model data, options, saved generated subsets, etc. default={logD['experiment_directory']}")



opt = vars(parser.parse_args())
  

main(opt)
