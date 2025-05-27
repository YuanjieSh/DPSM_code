import pandas as pd
import numpy as np
import math 
import sys
from scipy.stats.mstats import mquantiles
from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
#import torchsort
#from torchsort import soft_rank, soft_sort
import copy
import os 
from tqdm.autonotebook import tqdm
#from conformal_learning.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from conformal_learning.vgg import vgg16
from conformal_learning.densenet import densenet

import pickle 
import pdb
from conformal_learning.utils import *
from conformal_learning.help import *

from conformal_learning.losses import LDAMLoss, FocalLoss
from torch.utils.data.distributed import DistributedSampler
from conformal_learning.resnet import resnet
from conformal_learning.vgg import vgg16, vgg19_bn
from conformal_learning.densenet import densenet

from conformal_learning.sorting_nets import comm_pattern_batcher
from conformal_learning.variational_sorting_net import VariationalSortingNet
from conformal_learning.help import get_sos
from conformal_learning.smooth_conformal_prediction import smooth_aps_score, smooth_aps_score_all

from train.cifar100ManualData import load_cifar100
from train.Cal101ManualData import load_caltech101
from train.iNaturalistManualData import load_inaturalist

import jax
import gc
import os
os.environ['JAX_NUMPY_DTYPE_PROMOTION'] = 'relaxed'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.3"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# rng = jax.random.PRNGKey(42) 


def save_plot(ags, scores_return, path, param_taus, data_taus):
  sns.set_style("darkgrid")
  plt.plot(torch.sort(scores_return.detach().cpu()).values, linewidth = 3, color = 'r')
  #print(param_taus)
  #print(data_taus)
  plt.hlines(y = param_taus[-1], xmin = 0, xmax = len(scores_return), colors = 'lime', label = 'From training')
  plt.hlines(y = data_taus[-1], xmin = 0, xmax = len(scores_return), colors = 'blue', label = 'From data')
  plt.xlabel('Number of data points', fontsize = 25)
  plt.ylabel('{} Scores'.format(ags.train_CP_score), fontsize = 25)
  plt.savefig(path, dpi = 100, bbox_inches='tight', pad_inches=.1)

  plt.show()
  plt.close('all')   
  
  
# def loss_cal(y, y_pred, alpha = 0.1):
#   device = y.device
#   l1 = (1 - alpha) * (y - y_pred.to(device))
#   l2 = alpha * (y_pred.to(device) - y)
#   l1 = torch.relu(l1)
#   l2 = torch.relu(l2)
#   loss = torch.mean(l1 + l2)
#   return loss

def loss_cal(y, y_pred, alpha=0.1):
  # Ensure y and y_pred are on the same device
  device = y.device
  y_pred = y_pred.to(device)
    
  # Compute the loss
  loss = torch.where(
    y_pred >= y,
    (1 - alpha) * (y_pred - y),  # Case S >= q
    alpha * (y - y_pred)         # Case S < q
  )
    
  # Return the mean of the loss
  return torch.mean(loss)


class PinballMarginal(torch.nn.Module):
  def __init__(self, ):
    super().__init__()
    self.y_pred = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
  
  def forward(self, y, tau = False):
    if tau:
      return self.y_pred
    else:
      return loss_cal(y, self.y_pred)
  
class PinballClass(torch.nn.Module):
  def __init__(self, num_classes = 100):
    super().__init__()
    self.y_pred = torch.nn.Parameter(torch.rand(num_classes), requires_grad=True)
  
  def forward(self, S, Y):
    device = Y.device
    q = self.y_pred.to(device)[Y]
    return loss_cal(S, q)
  
  
  

class ConfDataset(Dataset):
    def __init__(self, dataset_XY, Z):
        self.dataset_XY = dataset_XY
        self.Z = Z
    
    def __getitem__(self, index):
        X, Y = self.dataset_XY[index]
        Z = self.Z[index]
        return X, Y, Z
      
    def __len__(self):
        return self.Z.shape[0]


def find_scores_APS(train_proba, y_train_batch, device):
  return compute_scores_diff(train_proba, y_train_batch, device = device)

def find_scores_RAPS(train_proba, y_train_batch, device):
  return compute_scores_diff_RAPS(train_proba, y_train_batch, device = device)

def find_scores_HPS(train_proba, y_train_batch, device):
  return compute_HPS_scores(train_proba, y_train_batch, device = device)

  
def Scores_APS_all_diff(probabilities:torch.tensor)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)
  n, K = proba_values.shape

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]

  # Sorting and ranking
  ranks_array_t = soft_rank(-proba_values, device = device, regularization_strength=REG_STRENGTH)-1
  prob_sort_t = -soft_sort(-proba_values, device = device, regularization_strength=REG_STRENGTH)

  ranks_array_t = ranks_array_t.to(device)
  prob_sort_t = prob_sort_t.to(device)

  # Compute the CDF
  Z_t = prob_sort_t.cumsum(dim=1)

  prob_cum_t = soft_indexing_all_labels(Z_t, ranks_array_t)
  # Compute the PMF of the observed labels
  # Compute the conformity scores
  U = torch.rand((n, K) ,dtype=float,device=device)

  APS_differentiable_score = prob_cum_t - proba_values * U
  return APS_differentiable_score 

def Scores_RAPS_all_diff(probabilities:torch.tensor, lambda_RAPS = 0.01, k_RAPS = 5)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)
  n, K = proba_values.shape

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]

  # Sorting and ranking
  ranks_array_t = soft_rank(-proba_values, device = device, regularization_strength=REG_STRENGTH)-1
  prob_sort_t = -soft_sort(-proba_values, device = device, regularization_strength=REG_STRENGTH)

  ranks_array_t = ranks_array_t.to(device)
  prob_sort_t = prob_sort_t.to(device)

  # Compute the CDF
  Z_t = prob_sort_t.cumsum(dim=1)

  prob_cum_t = soft_indexing_all_labels(Z_t, ranks_array_t)
  # Compute the PMF of the observed labels
  # Compute the conformity scores
  U = torch.rand((n, K) ,dtype=float,device=device)

  APS_differentiable_score = prob_cum_t - proba_values * U
  
  reg_term = torch.maximum(lambda_RAPS * (ranks_array_t.to(device) - k_RAPS), torch.zeros(ranks_array_t.shape).to(device))
  
  #print(reg_term.shape, ranks_array_t.shape)
  #exit(1)
  
  RAPS_differentiable_score = APS_differentiable_score +  reg_term
  
  return RAPS_differentiable_score 

def Scores_HPS_all_diff(probabilities:torch.tensor)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]


  return 1 - proba_values 
  
# Conformal Loss function
class UniformMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x, device, CDF_shift = 'KS', alpha = 0.1):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH, device = device)
    i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
    x_sorted = x_sorted.to(device)

    if CDF_shift == 'KS':
      out = torch.max(torch.abs(i_seq - x_sorted))
    elif CDF_shift == 'CM':
      out = torch.mean((i_seq - x_sorted)**2)
    elif CDF_shift == 'pinball':
      diff = x_sorted - i_seq
      diff_q1 = -alpha*diff
      diff_q2 = (1-alpha)*diff
      mask = (diff_q1 > diff_q2).int()
      out = torch.mean(mask * diff_q1 + (1 - mask) * diff_q2)
    return out

class PinballLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, theta, S, alpha):
    l1 = (1 - alpha) * (S - theta)
    l2 = alpha * (theta - S)

    return torch.relu(l1) + torch.relu(l2)
  
class ClassPinballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, num_classes=100, alpha=0.1):    
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()
    self.alpha = alpha
    self.nc = num_classes
    self.g = torch.nn.Parameter(torch.randn(num_classes) * 0.1 + (1 - alpha), requires_grad=True)
    self.pinball = PinballLoss()

  def forward(self, x, y, device):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    gY = torch.sum(torch.nn.functional.one_hot(y, num_classes=self.nc).float() * self.g.to(device), dim=-1)
    out = torch.mean(self.pinball(gY, x, self.alpha))
    #print(f"gY = {gY}")

    return out, self.g
  

class MarginalPinballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, alpha=0.1):    
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()
    self.alpha = alpha
    self.g = torch.nn.Parameter(torch.tensor([1 - alpha])*0.5, requires_grad=True)
    self.pinball = PinballLoss()

  def forward(self, scores, device):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    out = torch.mean(self.pinball(self.g.to(device), scores, self.alpha))

    return out, self.g
  
class DataMarginalPinnballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, alpha=0.1):   

    super().__init__()
 
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    self.alpha = alpha
    self.pinball = PinballLoss()

  def forward(self, scores, q):
    """ Compute the loss
    Parameters
    ----------
    scores : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    out = torch.mean(self.pinball(q, scores, self.alpha))
    #print(f"gY = {gY}")

    return out
  
class ExponentialMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x, CDF_shift, alpha = 0.1):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)

    U = np.random.uniform(size = batch_size)
    lam = 0.1
    X_exp = -np.log(1 - (1 - np.exp(-lam)) * U) / lam
    i_seq = soft_sort(X_exp.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
    x_sorted = x_sorted.to(device)
    i_seq = i_seq.to(device)

    if CDF_shift == 'KS':
      out = torch.max(torch.abs(i_seq - x_sorted))
    elif CDF_shift == 'CM':
      out = torch.mean((i_seq - x_sorted)**2)
    elif CDF_shift == 'pinball':
      diff = x_sorted - i_seq
      diff_q1 = -alpha*diff
      diff_q2 = (1-alpha)*diff
      mask = (diff_q1 > diff_q2).int()
      out = torch.mean(mask * diff_q1 + (1 - mask) * diff_q2)
    return out

class ChiSquareMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
    
    #i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
    x_sorted = x_sorted.to(device)

    out = torch.max(torch.abs(i_seq - x_sorted))
    return out

# def load_checkpoint(ags, model, path):
#   #print(path)
#   print(f"Loading the base model in order to fine tune")
#   if ags.arc == 'densenet100' or model_name == 'resnet110':
#     checkpoint = torch.load(path)
#     state_dict = checkpoint

#     # Remove only 'fc.weight' and 'fc.bias' from the state_dict
#     state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
#     state_dict.pop('fc.bias', None)
 
#     model.load_state_dict(state_dict)
#   else:
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint)
#   return model

# def load_checkpoint(ags, path):
#   #print(path)
#   print(f"Loading the base model in order to fine tune")
#   checkpoint = torch.load(path)
#   model.load_state_dict(checkpoint)

#   return model

def load_checkpoint(model, path):
  #print(path)
  print(f"Loading the base model in order to fine tune")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint)

  # filtered_state_dict = {k: v for k, v in checkpoint.items() if "running_mean" not in k and "running_var" not in k}
  # model.load_state_dict(filtered_state_dict, strict=False)  # Allow missing keys
  return model
        
def loss_fnc(train_rule = 'None', cls_num_list = None, num_epochs = 100, loss_type = 'CE'):
  
  if train_rule == 'None':
      #train_sampler = None  
      per_cls_weights = None 
  elif train_rule == 'Resample':
      #train_sampler = ImbalancedDatasetSampler(train_dataset)
      per_cls_weights = None
  elif train_rule == 'Reweight':
      #train_sampler = None
      beta = 0.9999
      effective_num = 1.0 - np.power(beta, cls_num_list)
      per_cls_weights = (1.0 - beta) / np.array(effective_num)
      per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
      per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)#.cuda(args.gpu)
  elif train_rule == 'DRW':
      #train_sampler = None
      idx = num_epochs // 1600
      betas = [0, 0.9999]
      effective_num = 1.0 - np.power(betas[idx], cls_num_list)
      per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
      per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
      per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)#.cuda(args.gpu)
  else:
      raise('Sample rule is not listed')

  if loss_type == 'CE':
      criterion_pred = nn.CrossEntropyLoss(weight=per_cls_weights) #.to(device)
  elif loss_type == 'LDAM':
      criterion_pred = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights) #.to(device)
  elif loss_type == 'Focal':
      criterion_pred = FocalLoss(weight=per_cls_weights, gamma=1) #.to(device)
  else:
      raise('Loss type is not listed')
    
  return criterion_pred
          
def load_state(model, path = None):
  path = path
  #try:
  #  checkpoint = torch.load(path)
  #except FileNotFoundError:
  #  path = input('Please give the correct path to load the model.')
  #  checkpoint = torch.load(path)
  #except IsADirectoryError:
  #  path = input('Please give the correct path to load the model.')
  
  checkpoint = torch.load(path)    
  state_dict = checkpoint

  # # Remove only 'fc.weight' and 'fc.bias' from the state_dict
  # state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
  # state_dict.pop('fc.bias', None)
 
  model.load_state_dict(state_dict)
  # model.load_state_dict(checkpoint)
  return model
          
def load_train_objs(ags, model_name = 'resnet20', method_name = 'Baseloss', lr = 0.1, optimizer_name = 'SGD'):
  
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=False).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=False).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=False).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  # Use pre-built torchvision model
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes)  # Adjust the fully connected layer
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
    
  wd = ags.finetune_weight_decay if ags.finetune else ags.base_weight_decay
  momentum = ags.finetune_momentum if ags.finetune else ags.base_momentum

  if method_name == 'pinball-class' or method_name == 'pinball-marginal' or method_name == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2' or method_name == 'pinball_marginal_with_Inefficiency' or method_name == 'pinball_class_with_Inefficiency':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'IW_test' or method_name == 'IW_Inefficiency' or method_name == 'only_Inefficiency' or method_name == 'only_test' or method_name == 'CPL' or method_name == 'CPL_test':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'Baseloss':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'Conformal' or method_name == 'only_CDF_gap':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
      
  elif method_name == 'only_pinball_marginal':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(list(model.parameters()) + list(pinball_model.parameters()), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(list(model.parameters()) + list(pinball_model.parameters()), lr=lr, momentum=momentum, weight_decay = wd)            
            
  return model, optimizer      

# def test_model(ags, model_name = 'vgg', path = None, previous = False):
#   if model_name == 'resnet110':
#     model = resnet(depth = 110, num_classes=ags.num_classes).cuda()
#   elif model_name == 'resnet20':
#     model = resnet(depth = 20, num_classes=ags.num_classes).cuda()
    
#   elif model_name == 'vgg':
#     model = vgg16().cuda()
#   elif model_name == 'vgg19_bn':
#     model = vgg19_bn().cuda()
#   elif model_name == 'densenet100':
#     model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2).cuda()
#   elif model_name == 'densenet161':
#     model = models.densenet161(pretrained=False)  
#     model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
#     model = model.cuda()
#   elif model_name == 'resnet18':
#     model = models.resnet18(pretrained=False)  
#     model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
#     model = model.cuda()
#   else:
#     raise("Please specify a correct model")
  
#   if model_name == 'densenet100' or model_name == 'resnet110':
#     checkpoint = torch.load(path)
#     state_dict = checkpoint

#     # Remove only 'fc.weight' and 'fc.bias' from the state_dict
#     state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
#     state_dict.pop('fc.bias', None)
 
#     model.load_state_dict(state_dict)
#     # model.load_state_dict(checkpoint)
    
#   else:
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint)
    
#   return model 

def test_model(ags, model_name = 'vgg', path = None, previous = False, use_fc_single = False):
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=use_fc_single).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint)
    
  return model 


def test_model_epoch(ags, model_name = 'vgg', path = None, previous = False, use_fc_single = False):
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=use_fc_single).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['MODEL_STATE'], strict=True)
    
  return model 

def prepare_dataloader(train_sample_dataset:Dataset, hout_sample_dataset:Dataset, batch_size:int):

  train_loader = torch.utils.data.DataLoader(
    train_sample_dataset, 
    batch_size=batch_size, 
    shuffle=False, drop_last=True, 
    sampler = DistributedSampler(train_sample_dataset))
  
  hout_loader = torch.utils.data.DataLoader(hout_sample_dataset, 
                batch_size=batch_size, shuffle=False, 
                sampler = DistributedSampler(hout_sample_dataset))


  return train_loader, hout_loader

def create_folder(ags, value):
  
  
  if ags.method == 'Baseloss':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/\
      gamma={}/weight_decay={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay)
    
  elif ags.method == 'pinball-class' or ags.method == 'pinball-marginal' or ags.method == 'pinball_marginal_with_Inefficiency' or ags.method == 'pinball_class_with_Inefficiency':
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/lr_qr={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.lr_qr, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'pinball_marginal_test':
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_qr={}/lr_qr={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.mu_qr, ags.lr_qr, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'pinball_marginal_test_2' :
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_qr={}/lr_qr={}/qr_decay_a={}/qr_decay_b={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.mu_qr, ags.lr_qr, ags.qr_decay_a, ags.qr_decay_b, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    


  elif ags.method == 'Conformal':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu={}/mu_s={}/mu_p={}/train_CP_score={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu, 0.0, 0.0, ags.train_CP_score)    

  elif ags.method == 'only_CDF_gap':
      file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu={}/mu_s={}/mu_p={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.train_rho, ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu, 0.0, 0.0, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T)    
    

  elif ags.method == 'IW_test' or ags.method == 'IW_Inefficiency' or ags.method == 'only_Inefficiency' or ags.method == 'only_test':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay,\
              ags.finetune, ags.mu_size, ags.mu_class, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs, ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule,\
              ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T)    


  elif ags.method == 'only_pinball_marginal':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T)    
  
  elif ags.method == 'CPL':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/lr_h={}/lr_lamda={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, \
              ags.finetune, ags.mu_size, ags.mu_class, ags.lr_h, ags.lr_lamda, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'CPL_test':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_lamda={}/lr_h={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, \
              ags.finetune, ags.mu_size, ags.mu_class, ags.mu_lambda, ags.lr_h, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    



  file_final = './ALLMODELS/AllModels_1/{}/final'.format(value)+file_name
  log_file = './ALLMODELS/AllModels_1/{}/logfile'.format(value)+file_name

  # file_name = file_name.replace(' ', '_')  # Replace problematic characters
  # file_final = os.path.abspath(f'./ALLMODELS/AllModels_1/{value}/final{file_name}')
  # log_file = os.path.abspath(f'./ALLMODELS/AllModels_1/{value}/logfile{file_name}')

  
  if not os.path.exists(file_final):
    
    try:
      os.makedirs(file_final)
    except FileExistsError:
      print('Already exist the main file\n')

  if not os.path.exists(log_file):
    
    try:
      os.makedirs(log_file)
    except FileExistsError:
      print('Already exist the log file\n')

  return file_final, log_file, file_name



def base_path_for_finetune(ags):
  
  file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/\
      gamma={}/weight_decay={}/".format(
              ags.data, ags.arc, 'CE', ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay)
      
  file_final = './ALLMODELS/AllModels_1/(Baseloss-check)/final'+file_name  

  return file_final

def create_final_data(ags, num_classes = 101):

  if ags.data == 'cifar100':
    _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_cifar100(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,num_classes=num_classes)

  elif ags.data == 'caltech':
     _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_caltech101(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,path='./data/caltech101',num_classes=num_classes)

  elif ags.data == 'inaturalist':
     _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_inaturalist(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,path='./data/inaturalist', num_classes=num_classes)
    
  if ags.method == 'Conformal':
    value = str('({}-check-1/5)'.format(ags.method))

  else:
    value = str('({}-check)'.format(ags.method))

    
  return train_dataset, val_dataset, value

def check_path(ags, file_final):
  
  if not ags.method == 'Baseloss':
    

    if not os.path.exists(file_final + '/final_epoch={}.pt'.format(ags.finetune_epochs)) and os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Base model exists but fine tuned model does not exist.')
      return True
    elif os.path.exists(file_final + '/final_epoch={}.pt'.format(ags.finetune_epochs)) and os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Both finetune and basemodel exists')
      return False



  else:
    if not os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Base model does not exist and need to train.')
      return True
    elif os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)):
      print('Base model exists and does not need to train.')
      return False    


@torch.no_grad()
def Estimate_quantile_n(model, dataloader, num_classes, device, alpha, score_name):
  _, logits, labels = get_logits_targets(model, dataloader, num_classes, device)

  n = len(logits)
  #print(f'n = {n}')
  ind = int(np.ceil((1 - alpha)*(1 + n)))
  
  soft_proba = torch.nn.Softmax(dim = 1)
  softmax_scores = soft_proba(logits)
  
  if score_name == 'APS':

    sos = get_sos(num_classes)

    scores = smooth_aps_score(softmax_scores, labels, sos=sos, device =device, dispersion=0.1, rng=None)

    sort_scores = torch.sort(scores).values
  
    return sort_scores[ind]    
    
  elif score_name == 'HPS':
    scores = 1 - softmax_scores
    scores = scores[np.arange(n), labels]
    sort_scores = torch.sort(scores).values
    return sort_scores[ind]     

  elif score_name == 'RAPS':
    scores = get_RAPS_scores_all(softmax_scores)
    scores = scores[np.arange(n), labels]
    sort_scores = torch.sort(scores).values
    return sort_scores[ind]   
  
  
  else:
    raise('score error')
    

def create_optimizers(model, optimizer, classification_lr=1e-3, scale_lr=1e-3):
  """
  Create optimizers for classification and scale prediction heads.
    
  Args:
    model: The backbone model (DenseNet, ResNet, etc.) with `fc` and `fc_single` layers.
    classification_lr: Learning rate for the classification head and shared layers.
    scale_lr: Learning rate for the scale prediction head and shared layers.
    
  Returns:
    optimizer_fc: Optimizer for the classification head.
    optimizer_fc_single: Optimizer for the scale prediction head.
  """
  # Get shared parameters
  shared_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc" not in name and "fc_single" not in name  # Exclude heads
  ]
    
  # Get classification head parameters (fc)
  classification_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc" in name and "fc_single" not in name
  ]
    
  # Get scale prediction head parameters (fc_single)
  scale_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc_single" in name
    ]

  # Define optimizers
  optimizer_fc = type(optimizer)(shared_params + classification_params, lr=classification_lr)
  optimizer_fc_single = type(optimizer)(shared_params + scale_params, lr=scale_lr)

  return optimizer_fc, optimizer_fc_single
  
  
  