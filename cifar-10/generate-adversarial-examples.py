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
test_batch = "cifar.python/cifar-10-batches-py/test_batch"
meta = "cifar.python/cifar-10-batches-py/batches.meta"
destination = "adversarial/"
dataset_max = 1000

args = argparse.ArgumentParser().parse_args()

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


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (32, 32))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


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

    def generate(self, original_image, im_label, label_names, destination, network_type):
        # I honestly dont know a better way to create a variable with specific value
        
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        # Convert to float tensor
        processed_image = preprocess_image(original_image)
        # Set the initial label after the forward propagation
        im_label_as_var = Variable(torch.from_numpy(np.asarray([im_label])))
        
        # Start iteration
        for i in range(10):
            print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            #processed_image.permute(0,2,3,1) 
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise
            
            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            #print(prep_confirmation_image.shape)
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.topk(1, 1, True, True)
            confirmation_prediction = confirmation_prediction[0][0]
            # Get Probability
            confirmation_confidence = nn.functional.softmax(confirmation_out, dim=1).data[0][confirmation_prediction]
            # Check if the prediction is different than the original
            if confirmation_prediction != im_label:
                print("Generated adversarial eg against %s" %(network_type))
                print('Original image was predicted as:', label_names[im_label].decode("utf-8"),
                      'with adversarial noise converted to:', label_names[confirmation_prediction].decode("utf-8"),
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                
                # Create file names
                name_end = label_names[im_label].decode("utf-8") + "_to_" + label_names[confirmation_prediction].decode("utf-8") + ".jpg"
                noise_name = "noise_from_" + name_end
                image_name = "img_from_" + name_end

                # Save the image into the 0, 100 and noise folder
                img_0_path = os.path.join(destination, "0-" + network_type, label_names[im_label].decode("utf-8"), image_name)
                img_100_path = os.path.join(destination, "100-" + network_type, label_names[confirmation_prediction].decode("utf-8"), image_name)
                noise_path = os.path.join(destination, "noise-" + network_type, label_names[im_label].decode("utf-8"), noise_name)

                cv2.imwrite(noise_path, noise_image)
                
                cv2.imwrite(img_0_path, recreated_image)
                cv2.imwrite(img_100_path, recreated_image)
                break

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
    

    dataset = unpickle(test_batch)
    label_names = unpickle(meta)[b'label_names']
    dataset_size = len(dataset[b'data'])
    dataset_size = dataset_max if dataset_size > dataset_max else dataset_size

    FGS_untargeted_full = FastGradientSignUntargeted(net_full, 0.01)
    FGS_untargeted_twn = FastGradientSignUntargeted(net_twn, 0.01)

    # create paths for 0-twn, 100-twn, 0-full, 100-full, noise-twn, noise-full
    make_label_directory(os.path.join(destination, "clean"), label_names)
    make_label_directory(os.path.join(destination, "0-twn"), label_names)
    make_label_directory(os.path.join(destination, "100-twn"), label_names)
    make_label_directory(os.path.join(destination, "0-full"), label_names)
    make_label_directory(os.path.join(destination, "100-full"), label_names)
    make_label_directory(os.path.join(destination, "noise-twn"), label_names)
    make_label_directory(os.path.join(destination, "noise-full"), label_names)
    
    for image_counter in range(0, dataset_size):
        # loop through all the images in the dataset and transposing it
        original_image = dataset[b'data'][image_counter,:]
        correct_label = dataset[b'labels'][image_counter]
        original_image = original_image.reshape(3,32,32).transpose([1, 2, 0])
        # perform forward propagation that we work with images which the network is predicting correctly
        # Process image
        # Convert to float tensor
        processed_image = preprocess_image(original_image)
        # Forward pass
        initial_out = net_full(processed_image)
        # Get prediction
        initial_prediction = initial_out.data.topk(1, 1, True, True)[1][0][0]
        print("\nData label: %d\tModel output: %d" %(correct_label, initial_prediction))

        if (initial_prediction != correct_label):
            print("Mismatch labels")
            continue

        print("Performing adversarial data generation")
        # save the clean image into the right labels
        image_name = str(image_counter) + ".jpg"
        clean_path = os.path.join(destination, "clean", label_names[correct_label].decode("utf-8"), image_name)
        cv2.imwrite(clean_path, original_image)
        # create targeted examples against full precision networks
        FGS_untargeted_full.generate(original_image, correct_label, label_names, destination, "full")
        # create targeted examples against ternary networks
        FGS_untargeted_twn.generate(original_image, correct_label, label_names, destination, "twn")


main()