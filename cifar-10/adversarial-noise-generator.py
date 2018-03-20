# Adversarial
import _init_paths
import copy
import cv2
import os
import shutil
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models
import pickle
from PIL import Image
import numpy as np

# HYPERPARAMETERS
# OVERALL VARIABLES FOR THE PROGRAM
#test_batch = "cifar.python/cifar-10-batches-py/test_batch"
meta = "cifar.python/cifar-10-batches-py/batches.meta"
destination = "adversarial/"
dataset_max = 10000

args = argparse.ArgumentParser().parse_args()

args.image_folder_clean = "images/clean"
# FULL WEIGHTS
args.pretrained_full = "weights/cifar10-full/model_best.pth.tar"
# TERNARY WEIGHTS
args.pretrained_twn = "weights/cifar10-twn/model_restored.pth.tar"
# Other variables
args.data_path = "cifar.python"
args.arch = "resnet20"
args.batch_size = 1
args.workers = 16
args.ngpu = 0

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self, input_var, target_var, label_names, destination, network_type, mean_sum, img_counter):
        # I honestly dont know a better way to create a variable with specific value
        # Inverse normalization for saving
        inv_mean = [x / 255 for x in [-125.3, -123.0, -113.9]]
        inv_std = [255 / x for x in [63.0, 62.1, 66.7]]
        zero_mean = [0 for x in [-125.3, -123.0, -113.9]]
        one_std = [1 for x in [63.0, 62.1, 66.7]]
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        im_label = target_var.data[0]

        original_var = input_var.clone()
        
        # Start iteration
        for i in range(100):
            #print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            input_var.grad = None
            # Forward pass
            out = self.model(input_var)
            # Calculate CE loss
            pred_loss = ce_loss(out, target_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(input_var.grad.data)
            # Add Noise to processed image
            input_var.data = input_var.data + adv_noise
            confirmation_out = self.model(input_var)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.topk(1, 1, True, True)
            confirmation_prediction = confirmation_prediction[0][0]
            # Get Probability
            confirmation_confidence = nn.functional.softmax(confirmation_out, dim=1).data[0][confirmation_prediction]
            # Check if the prediction is different than the original
            if (confirmation_prediction != im_label):
                '''
                print("Generated adversarial eg against %s" %(network_type))
                print('Original image was predicted as:', label_names[im_label].decode("utf-8"),
                      'with adversarial noise converted to:', label_names[confirmation_prediction].decode("utf-8"),
                      'and predicted with confidence of:', confirmation_confidence)
                      '''

                # Create file names
                name_end = label_names[im_label].decode("utf-8") + "_to_" + label_names[confirmation_prediction].decode("utf-8") + str(img_counter) + ".jpg"
                image_name = "img_from_" + name_end

                # Save the image into the 0, 100
                img_0_path = os.path.join(destination, "0-" + network_type, label_names[im_label].decode("utf-8"), image_name)
                img_100_path = os.path.join(destination, "100-" + network_type, label_names[confirmation_prediction].decode("utf-8"), image_name)

                # Generate clean image for saving
                tensor = transforms.Normalize(zero_mean, inv_std)(input_var.data[0])
                tensor = transforms.Normalize(inv_mean, one_std)(tensor)
                original_image = transforms.ToPILImage(mode=None)(tensor)
                original_image.save(img_0_path)
                original_image.save(img_100_path)

                # Calculate the difference in noise level
                diff = (100 * input_var - 100 * original_var)
                torch_mean = torch.mean(torch.abs(diff)).data[0]
                mean_sum = mean_sum + torch_mean

                return mean_sum
        return 1

def make_label_directory(directory, label_names):
    # Delete existing data and create the directory
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    # Loop through all the labels and create the directory
    for label in label_names:
        label_str = label.decode("utf-8")
        label_path = os.path.join(directory, label_str)
        os.makedirs(label_path)

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

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    inv_mean = [x / 255 for x in [-125.3, -123.0, -113.9]]
    inv_std = [255/x for x in [63.0, 62.1, 66.7]]
    zero_mean = [0 for x in [-125.3, -123.0, -113.9]]
    one_std = [1 for x in [63.0, 62.1, 66.7]]

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    path = os.path.join(args.image_folder_clean)
    #test_data = dset.ImageFolder(path, transform=test_transform)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("Loading dataset")
    label_names = unpickle(meta)[b'label_names']

    FGS_untargeted_full = FastGradientSignUntargeted(net_full, 0.001)
    FGS_untargeted_twn = FastGradientSignUntargeted(net_twn, 0.001)

    # create paths for 0-twn, 100-twn, 0-full, 100-full, noise-twn, noise-full
    make_label_directory(os.path.join(destination, "clean-twn"), label_names)
    make_label_directory(os.path.join(destination, "clean-full"), label_names)
    make_label_directory(os.path.join(destination, "0-twn"), label_names)
    make_label_directory(os.path.join(destination, "100-twn"), label_names)
    make_label_directory(os.path.join(destination, "0-full"), label_names)
    make_label_directory(os.path.join(destination, "100-full"), label_names)

    full_counter = 0
    twn_counter = 0
    full_sum = 0.0 
    twn_sum = 0.0

    for i, (input_img, target) in enumerate(test_loader):
        if (i > dataset_max):
            break
        print("Image:", str(i))

        input_var = torch.autograd.Variable(input_img, requires_grad=True)
        target_var = torch.autograd.Variable(target)
        im_label = target_var.data[0]
        full_out = net_full(input_var)
        twn_out = net_twn(input_var)
        # Generate clean image for saving
        tensor = transforms.Normalize(zero_mean, inv_std)(input_var.data[0])
        tensor = transforms.Normalize(inv_mean, one_std)(tensor)
        original_image = transforms.ToPILImage(mode=None)(tensor)
        image_name = str(i) + ".jpg"
        # Get prediction
        full_prediction = full_out.data.topk(1, 1, True, True)[1][0][0]
        twn_prediction = twn_out.data.topk(1, 1, True, True)[1][0][0]
        print("\nData label: %d\tFull: %d\tTwn: %d" %(im_label, full_prediction, twn_prediction))

        if (full_prediction == im_label):
            print("Generating adversarial for full")
            clean_path = os.path.join(destination, "clean-full", label_names[im_label].decode("utf-8"), image_name)
            original_image.save(clean_path)
            # create targeted examples against full precision networks
            full_sum = FGS_untargeted_full.generate(input_var, target_var, label_names, destination, "full", full_sum, i)
            full_counter = full_counter + 1

        if (twn_prediction == im_label):
            print("Generating adversarial for twn")
            clean_path = os.path.join(destination, "clean-twn", label_names[im_label].decode("utf-8"), image_name)
            original_image.save(clean_path)
            # create targeted examples against full precision networks
            twn_sum = FGS_untargeted_twn.generate(input_var, target_var, label_names, destination, "twn", twn_sum, i)
            twn_counter = twn_counter + 1    

    print("Full Counter:", full_counter)
    print("TWN Counter:", twn_counter)
    print("Full Mean Diff:", full_sum / full_counter)
    print("TWN Mean Diff:", twn_sum / twn_counter)

main()