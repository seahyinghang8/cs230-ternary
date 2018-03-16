import _init_paths
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models

args = argparse.ArgumentParser().parse_args()

args.arch = "resnet20"
args.pretrained = "weights/cifar10-full/model_best.pth.tar"
#args.pretrained = "weights/cifar10-twn/model_restored.pth.tar"
args.data_path = "cifar.python"
args.batch_size = 128
args.workers = 16
args.ngpu = 0

def main(args):
  num_classes = 10

  mean = [x / 255 for x in [125.3, 123.0, 113.9]]
  std = [x / 255 for x in [63.0, 62.1, 66.7]]

  test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
  # DEFAULT TEST DATA LOADER
  test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=False)
  # IMAGE TEST DATA LOADER
  #test_data = dset.ImageFolder("images", transform=test_transform)
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