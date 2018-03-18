
# coding: utf-8

# In[1]:


# Adversarial
import _init_paths
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models
import easydict
# Noise Generator
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc

# Ensure plots embeded in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#args = argparse.ArgumentParser().parse_args(argv[1:])
# args = argparse.ArgumentParser().parse_args()
# args.arch = "resnet20"
# args.pretrained = "weights/cifar10-full/model_best.pth.tar"
# #args.pretrained = "weights/cifar10-twn/model_restored.pth.tar"
# args.data_path = "cifar.python"
# args.batch_size = 128
# args.workers = 16
# args.ngpu = 0


# In[16]:


args = easydict.EasyDict({
        "arch" : "resnet20",
        "pretrained" : "weights/cifar10-full/model_best.pth.tar",
        "data_path" : "cifar.python",
        "batch_size" : 128,
        "workers" : 16,
        "ngpu": 0
})

def main(args):
  num_classes = 10

  mean = [x / 255 for x in [125.3, 123.0, 113.9]]
  std = [x / 255 for x in [63.0, 62.1, 66.7]]

  test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  # DEFAULT TEST DATA LOADER
  #test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=False)
  # IMAGE TEST DATA LOADER
  test_data = dset.ImageFolder("images", transform=test_transform)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
  
  print("=> creating model '{}'".format(args.arch))
  # Init model, criterion, and optimizer
  net = models.__dict__[args.arch](num_classes) #NOTE load the architecture  

  print("=> network :\n {}".format(net))
  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  pretrained = torch.load(args.pretrained, map_location={'cuda:0': 'cpu'}) #NOTE Load Pre-trained Model
  
  try:
    net.load_state_dict(pretrained['state_dict']) #NOTE reconstructs the model
  except KeyError as e:
    print(e)

  print("=> loaded pretrained model '{}'".format(args.pretrained))
  
  # NOTE ATTEMPT TO PRINT MODEL
  #for m in net.modules():
  #  if isinstance(m, torch.nn.Conv2d):
  #    print(m.weight)

  # run the validation program
  validate(test_loader, net)


#NOTE net -> model, val_loader is the data loader, citerion is just a comparator module
def validate(val_loader, model):
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  #NOTE input is a tensor batch of images, target is a tensor of labels
  for i, (input, target) in enumerate(val_loader):
    input_var = torch.autograd.Variable(input, volatile=True) #NOTE constructs a variable from the input
    target_var = torch.autograd.Variable(target, volatile=True) #NOTE

    # compute output
    output = model(input_var)

    # measure accuracy
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5)) #NOTE
    top1.update(prec1[0], input.size(0))
    top5.update(prec5[0], input.size(0))

  print('**Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

  return top1.avg

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

main(args)


# In[96]:


#unPickle
import pickle
#rePickle
from PIL import Image
import numpy as np
#Saving Images
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
        #print(normal_image.shape)
        #plt.imshow(normal_image)
    return normal_image

        
print("functions were defined")

file = "/Users/Behzad/CS230/Data/cifar-10-batches-py-original/test_batch"
# pickle.dump(dictionary, files)
# print("dumped")
# print(dictionary[b'data'])
# print(dictionary[b'data'].shape)
# print(dictionary[b'data'].shape)
loadImageFromDB(file, 0, plotImage=True)


# Now Let's work on the adversarial example generator

# In[5]:


"""
Created on Fri Dec 15 19:57:34 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

#from misc_functions import preprocess_image, recreate_image, get_params


# In[103]:


"""
Misc. Functions
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import copy
import cv2
import numpy as np

import torch
from torch.autograd import Variable
# from torchvision import models
import models.cifar as models

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


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
#     example_list = [['../input_images/apple.JPEG', 948],
#                     ['../input_images/eel.JPEG', 390],
#                     ['../input_images/bird.JPEG', 13]]
#     selected_example = example_index
#     img_path = example_list[selected_example][0]
    img_path = example_index
    #TODO: Add forward prop here.
    
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = cv2.imread(img_path, 1)
    cv2.imshow('original_image', original_image)
    # Process image
    #prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    print (pretrained_model)
    pretrained_model = torch.load(args.pretrained, map_location={'cuda:0': 'cpu'}) #NOTE Load Pre-trained Model
    print (pretrained_model)



    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


# In[133]:


import _init_paths
from models.resnet import resnet20, resnet32, resnet44

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

    def generate(self, original_image, im_label):
        # I honestly dont know a better way to create a variable with specific value
        
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
#       Process image
        # Convert to float tensor
        print(original_image.shape)
        print("Behzad is the new electricity")
        processed_image = preprocess_image(original_image)
        print(processed_image.shape)
        
        #im_as_ten = torch.from_numpy(original_image).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        #im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        #im_as_var = Variable(im_as_ten, requires_grad=True)
        #processed_image = im_as_var
        
        #--
        # zero_gradients(x)
        # Zero out previous gradients
        # Can also use zero_gradients(x)
        processed_image.grad = None
        # Forward pass
        #processed_image.permute(0,2,3,1) 
        out = self.model(processed_image)
        # Calculate CE loss
        
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
        _, confirmation_prediction = confirmation_out.data.max(1)
        # Get Probability
        confirmation_confidence =             nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
        # Convert tensor to int
        confirmation_prediction = confirmation_prediction.numpy()[0]
        im_label = confirmation_prediction
        #--
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
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence =                 nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            if confirmation_prediction != im_label:
                print('Original image was predicted as:', im_label,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                cv2.imwrite('../generated/untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                            str(confirmation_prediction) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('../generated/untargeted_adv_img_from_' + str(im_label) + '_to_' +
                            str(confirmation_prediction) + '.jpg', recreated_image)
                break

        return 1


if __name__ == '__main__':
    
    
    #loading the model
    num_classes = 10
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]] 
    net = resnet20(num_classes) #NOTE load the architecture  

    print("=> network :\n {}".format(net))
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    pretrained = torch.load(args.pretrained, map_location={'cuda:0': 'cpu'}) #NOTE Load Pre-trained Model

    try:
        net.load_state_dict(pretrained['state_dict']) #NOTE reconstructs the model
    except KeyError as e:
        print(e)

    print("=> loaded pretrained model '{}'".format(args.pretrained))
    
    databaseDir = "/Users/Behzad/CS230/Data/cifar-10-batches-py-original/data_batch_1"
    original_image = loadImageFromDB(databaseDir, 302, plotImage=True)
    
    print(original_image.shape)
    target_example = '/Users/Behzad/Downloads/ternary/cifar/images/airplane/airplane1.jpg'
#   (original_image, prep_img, target_class, _, pretrained_model) =\
#   get_params(target_example)

    FGS_untargeted = FastGradientSignUntargeted(net, 0.01)
    FGS_untargeted.generate(original_image, 8)


# In[90]:


# cv2.imshow('/Users/Behzad/Downloads/ternary/cifar/images/airplane/airplane1.jpg', 1)


# In[9]:


# example_index = '/Users/Behzad/Downloads/ternary/cifar/images/airplane/airplane1.jpg'
# img_path = example_index
# #TODO: Add forward prop here.
# target_class = 1
# file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
# # Read image
# original_image = cv2.imread(img_path, 1)
# cv2.imshow('original_image', original_image)

