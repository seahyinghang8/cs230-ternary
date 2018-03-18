# Evaluate both the ternary and full precision models against the noisy datasets
# Some imports are based of code from Zhu et al and thus they are not made public on github 
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
args.image_folder = "adversarial/"
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

def main(args):
  num_classes = 10

  mean = [x / 255 for x in [125.3, 123.0, 113.9]]
  std = [x / 255 for x in [63.0, 62.1, 66.7]]

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
  

  # run through all the test datasets
  test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  # DEFAULT TEST DATA LOADER
  #test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=False)
  # IMAGE TEST DATA LOADER
  image_list = os.listdir(args.image_folder)
  image_list.sort()
  for name in image_list:
    path = os.path.join(args.image_folder, name)
    test_data = dset.ImageFolder(path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("Loading dataset %s" %(name))
    print("Running Full Precision Model:")
    validate(test_loader, net_full)
    print("Running Ternary Model:")
    validate(test_loader, net_twn)
    print("\n")


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