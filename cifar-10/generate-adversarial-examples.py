# Adversarial
import _init_paths
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models

args = argparse.ArgumentParser().parse_args()

# IMAGE DATA FOLDER
args.image_folder = "images/"
# FULL WEIGHTS
args.pretrained_full = "weights/cifar10-full/model_best.pth.tar"
# TERNARY WEIGHTS
args.pretrained_twn = "weights/cifar10-twn/model_restored.pth.tar"
# Other variables
args.data_path = "cifar.python"
args.arch = "resnet20"
args.batch_size = 128
args.workers = 16
args.ngpu = 0

#unPickle
import pickle
from PIL import Image
import numpy as np
import scipy.misc

def unPickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def rePickle(dict, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict, file)

# Loads the requested image from the given database 
def loadImageFromDB (database, imageNumber, plotImage = False):
    dictionary = unPickle(file)
    normal_image= dictionary[b'data'][imageNumber,:]   
    if (plotImage == True):
        normal_image = normal_image.reshape(3,32,32).transpose([1, 2, 0])
    return normal_image

def main():
    # Main will load both the models and run adversarial examples
    num_classes = 10

    # Init model, criterion, and optimizer for both twn and full
    net_full = models.__dict__[args.arch](num_classes) #NOTE load the architecture  
    net_twn = models.__dict__[args.arch](num_classes) 

    net_full = torch.nn.DataParallel(net_full, device_ids=list(range(args.ngpu)))
    net_twn = torch.nn.DataParallel(net_twn, device_ids=list(range(args.ngpu)))

    pretrained_full = torch.load(args.pretrained_full, map_location={'cuda:0': 'cpu'}) #NOTE Load Pre-trained Model
    pretrained_twn = torch.load(args.pretrained_twn, map_location={'cuda:0': 'cpu'}) #NOTE Load Pre-trained Model
    
    try:
      net_full.load_state_dict(pretrained_full['state_dict']) #NOTE reconstructs the model
    except KeyError as e:
      print("Error - Full")
      print(e)

    try:
      net_twn.load_state_dict(pretrained_twn['state_dict']) #NOTE reconstructs the model
    except KeyError as e:
      print("Error - Ternary")
      print(e)
    
    module_num = 18
    counter = 0
    print("\nFull Precision Weights")
    for m in net_full.modules():
      if (counter == module_num):
        print(m.weight[0][0])
      counter += 1
    
    counter = 0
    print("Ternary Weights")
    for m in net_twn.modules():
      if (counter == module_num):
        print(m.weight[0][0])
      counter += 1
    
    databaseDir = "/Users/Behzad/CS230/Data/cifar-10-batches-py-original/data_batch_1"
    original_image = loadImageFromDB(databaseDir, 302, plotImage=True)
    
    print(original_image.shape)
    target_example = '/Users/Behzad/Downloads/ternary/cifar/images/airplane/airplane1.jpg'
#   (original_image, prep_img, target_class, _, pretrained_model) =\
#   get_params(target_example)

    FGS_untargeted = FastGradientSignUntargeted(net, 0.01)
    FGS_untargeted.generate(original_image, 8)