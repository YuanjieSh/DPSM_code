from datetime import date
import random
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle 
from tqdm import tqdm
import torchvision.transforms as T
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.distributed import init_process_group, destroy_process_group

import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split
import seaborn as sns
import math

import parser_file
from cifar100ManualData import load_cifar100

sys.path.insert(0, './')

from conformal_learning.losses import LDAMLoss, FocalLoss


from conformal_learning.utils import evaluate_predictions, get_scores_HPS, get_scores, classwise_conformal, Marginal_conformal
from conformal_learning import black_boxes_CNN
from conformal_learning.utils import *
from conformal_learning.help import *
from conformal_learning.black_boxes_CNN import Estimate_quantile_n, Scores_RAPS_all_diff, Scores_APS_all_diff, Scores_HPS_all_diff, PinballMarginal, UniformMatchingLoss, Estimate_size_loss_RAPS, save_plot, find_scores_RAPS, find_scores_APS, find_scores_HPS, load_train_objs, base_path_for_finetune, load_checkpoint, prepare_dataloader, loss_fnc, check_path, create_final_data, create_folder, test_model, loss_cal, create_optimizers
from conformal_learning.smooth_conformal_prediction import smooth_aps_score, smooth_aps_score_all


date = date.today().strftime("%m-%d-%Y")

import jax
# rng = jax.random.PRNGKey(42) 
import os
os.environ['JAX_NUMPY_DTYPE_PROMOTION'] = 'relaxed'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.3"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

# from torch.cuda.amp import autocast, GradScaler


# torch.autograd.set_detect_anomaly(True)

# torch.cuda.empty_cache()



    
def ddp_setup():
  init_process_group(backend = "nccl")

def cleanup_memory():
    torch.cuda.empty_cache()  
    torch.cuda.synchronize()  


          
class Trainer:
  def __init__(self,
               model:torch.nn.Module,
               cdf_gap:torch.nn.Module,
               train_data:DataLoader,
               val_data:DataLoader,
               optimizer:torch.optim.Optimizer,
               loss_fn: torch.nn.CrossEntropyLoss,
               mu: float,
               mu_size:float,
               mu_class:float,
               batch_size:int, 
               save_path:str,
               log_file:str,
               train_alpha:float,
               method: str,
               num_classes: int,
               save_every: int,
               ags:argparse,
               )->None:

    self.local_rank = int(os.environ["LOCAL_RANK"])
    self.global_rank = int(os.environ["RANK"])
    self.model = model.to(self.local_rank)
    self.train_data = train_data
    self.optimizer = optimizer
    #print('0000', self.optimizer.param_groups[0]['params'][-1])
    
    self.tau = torch.tensor(0.9, requires_grad=True).to(self.local_rank)
    self.z = torch.tensor(0.9, requires_grad=True).to(self.local_rank)
    # self.h = torch.tensor(0.9, requires_grad=True).to(self.local_rank)
    # self.lamda= torch.tensor(0.5, requires_grad=True).to(self.local_rank)
    self.lamda = nn.Parameter(torch.tensor(0.5, requires_grad=True).to(self.local_rank))

    self.val_data = val_data
    self.loss_fn = loss_fn
    self.mu = mu
    self.mu_size = mu_size
    self.mu_class = mu_class
    self.batch_size = batch_size,
    self.save_path = save_path

    self.log_file = log_file
    self.soft_proba = torch.nn.Softmax(dim = 1)
    self.train_alpha = train_alpha
    self.method = method
    self.save_every = save_every
    self.epochs_run = 0
    self.ags = ags
    self.size = len(self.train_data)
    self.T = self.ags.sigmid_T

    self.qr_decay = 1.0
    self.cdf_gap = cdf_gap
    self.num_classes = num_classes
    
    if os.path.exists(self.save_path+'/snapshot.pt'):
      #print(self.save_path+'/final_epoch={}.pt'.format(self.ags.finetune_epochs+1))
      #print(save_path)
      #print(ah)
      #print("Loading snapshot")
      self._load_snapshot(self.save_path+'/snapshot.pt')
      
    self.model = DDP(self.model, device_ids=[self.local_rank])

    if self.ags.method == 'CPL':
      self.optimizer_fc, self.optimizer_fc_single = create_optimizers(self.model, self.optimizer, classification_lr=self.ags.finetune_lr, scale_lr=self.ags.lr_h)
      # self.optimizer_h = type(self.optimizer)(self.h.parameters(), lr=self.ags.lr_h)
      # self.optimizer_lambda = type(self.optimizer)([self.lamda], lr=self.ags.lr_lamda)

    lr_milestones = self.ags.finetune_lr_schedule if self.ags.finetune else self.ags.base_lr_schedule
    gamma = self.ags.finetune_gamma if self.ags.finetune else self.ags.base_gamma
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=gamma)
    # self.scaler = GradScaler()
    

  def _load_snapshot(self, snapshot_path):
    snapshot = torch.load(snapshot_path)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run = snapshot["EPOCH"]
    if self.ags.classwise_training:
      print(f'Will begin class wise training from 0 epochs')
      self.epochs_run = 0
      
    print(f"Resuming the model training from epoch {self.epochs_run}")
    
    
  def _run_batch(self, source, targets):
    self.optimizer.zero_grad()
    output = self.model(source)
    loss = self.loss_fn(output, targets)
    loss.backward()
    self.optimizer.step()
    return loss
        

        
  def _run_batch_only_CDF_gap(self, source, targets):
    self.optimizer.zero_grad()
    output = self.model(source)
    
    """Add a temperature scaling T=1 in order to smooth the probability. Without scaling, the model gives very high probability for prediction that consists some noise.
    """
    Temp = self.ags.train_T
    output /= Temp
  

    if self.ags.train_CP_score == 'APS':
      scores = find_scores_APS(self.soft_proba(output), targets, device = self.local_rank)
    elif self.ags.train_CP_score == 'HPS':
      scores = find_scores_HPS(self.soft_proba(output), targets, device = self.local_rank)
    elif self.ags.train_CP_score == 'RAPS':
      scores = find_scores_RAPS(self.soft_proba(output), targets, device = self.local_rank)
      
    loss1 = self.cdf_gap(scores, device = self.local_rank)
    loss = self.ags.mu * loss1
    
    if self.ags.finetune_CE:
      #print('adding CE loss as well')
      loss_ce = self.loss_fn(output, targets)
      loss += loss_ce
      
    loss.backward()
    self.optimizer.step()
    
    self.scores_return  = scores
    
    if self.local_rank == 0:  
      q_data1 = torch.quantile(scores, q = 1-self.train_alpha).detach().item()
      self.data_taus.append(q_data1)
    return loss1

  def _run_batch_only_Inefficiency(self, source, targets):

    output = self.model(source)
    ind1, ind2 = train_test_split(torch.arange(len(output)), train_size=0.499, random_state=1111) 
    
    """Add a temperature scaling T=1 in order to smooth the probability. Without scaling, the model gives very high probability for prediction that consists some noise.
    """
    Temp = self.ags.train_T
    output /= Temp
    
    if self.ags.train_CP_score == 'HPS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      scores = get_HPS_scores(softmax_scores_1, targets[ind1])
      # scores_all = get_HPS_scores_all(softmax_scores_2)

      tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)

      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_HPS_hard(softmax_scores_2, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         

      # scores_all = Scores_HPS_all_diff(self.soft_proba(output[ind1]))

    elif self.ags.train_CP_score == 'APS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      
      sos = get_sos(self.num_classes)
      scores = smooth_aps_score(softmax_scores_1, targets[ind1], sos=sos, device =self.local_rank, dispersion=0.1, rng=None)
      # scores = compute_scores_diff(softmax_scores_1, targets[ind1], device = self.local_rank)
      # scores_all = get_APS_scores_all(softmax_scores_2, randomize=True)

      tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)

      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_APS_hard(softmax_scores_2, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         
      # scores_all = Scores_APS_all_diff(self.soft_proba(output[ind1]))

    elif self.ags.train_CP_score == 'RAPS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      scores = compute_scores_diff_RAPS(softmax_scores_1, targets[ind1], device = self.local_rank, lambda_RAPS =.01, k_RAPS =5)
      # scores_all = get_RAPS_scores_all(softmax_scores_2, lmbda=.01, kreg=5, randomize=True)

      tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)
      # scores_all = Scores_RAPS_all_diff(self.soft_proba(output[ind1]))
      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_RAPS_hard(softmax_scores_2, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         

    # sets1, sets2, sets3 = estimate_cardinality1(scores_all, qn = tau) #scores_all_ = (n, ), scores_all = (n, K)
    
    
    loss_class = self.mu_class  * class_loss
    
    loss_size = self.mu_size * soft_size_loss
    
    
    if self.ags.finetune_CE:
      #print('adding CE loss as well')
      loss_ce = self.loss_fn(output, targets)
      loss = loss_ce + loss_size + loss_class 
      
    #loss = self.mu_size * loss2

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.scores_return  = scores
    
    if self.local_rank == 0:  
      q_data1 = torch.quantile(scores, q = 1-self.train_alpha).detach().item()
      self.param_taus.append(self.tau.data)
      self.data_taus.append(q_data1)
      
      self.stats['loss_all'].append(loss.item())
      self.stats['loss_size'].append(loss_size.item())
      self.stats['class_size_loss'].append(loss_class.item())
      self.stats['CE_loss'].append(loss_ce.item())

      self.stats['scale_soft_set_size'].append(soft_size_loss.item())
      self.stats['indicator_hard_set_size'].append(hard_size_loss.item())
      self.eval_stats['Q_b_train'].append(tau)
      
      
    return


  def _run_batch_pinball_marginal_with_Inefficiency(self, source, targets):
    output = self.model(source)
    
    """Add a temperature scaling T=2 in order to smooth the probability. Without scaling, the model gives very high probability for prediction that consists some noise.
    """
    ind1, ind2 = train_test_split(torch.arange(len(output)), train_size=0.499, random_state=1111) 
    
    Temp = self.ags.train_T
    output /= Temp

    if self.ags.train_CP_score == 'HPS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      scores = get_HPS_scores(softmax_scores_1, targets[ind1])
      loss1 = loss_cal(self.tau, scores)
      # scores_all = get_HPS_scores_all(softmax_scores_2)

      # tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)

      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_HPS_hard(softmax_scores_2, targets[ind2], self.tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 0)         

      # scores_all = Scores_HPS_all_diff(self.soft_proba(output[ind1]))

    elif self.ags.train_CP_score == 'APS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      
      sos = get_sos(self.num_classes)
      scores = smooth_aps_score(softmax_scores_1, targets[ind1], sos=sos, device =self.local_rank, dispersion=0.1, rng=None)
      # scores = compute_scores_diff(softmax_scores_1, targets[ind1], device = self.local_rank)
      loss1 = loss_cal(self.tau, scores)
      # scores_all = get_APS_scores_all(softmax_scores_2, randomize=True)

      # tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)

      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_APS_hard(softmax_scores_2, targets[ind2], self.tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 0)         
      # scores_all = Scores_APS_all_diff(self.soft_proba(output[ind1]))

    elif self.ags.train_CP_score == 'RAPS':
      softmax_scores_1 = self.soft_proba(output[ind1])
      softmax_scores_2 = self.soft_proba(output[ind2])
      scores = compute_scores_diff_RAPS(softmax_scores_1, targets[ind1], device = self.local_rank, lambda_RAPS =.01, k_RAPS =5)
      loss1 = loss_cal(self.tau, scores)
      # scores_all = get_RAPS_scores_all(softmax_scores_2, lmbda=.01, kreg=5, randomize=True)

      # tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)
      # scores_all = Scores_RAPS_all_diff(self.soft_proba(output[ind1]))
      soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_RAPS_hard(softmax_scores_2, targets[ind2], self.tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         
        

    loss_class = self.mu_class * class_loss

    # print(loss_class)
    # print(loss_class.item())
    
    loss_size = self.mu_size * soft_size_loss

    # print(loss_size)
    # print(loss_size.item())

    # print(loss1)
    # print(loss1.item())
        
    if self.ags.finetune_CE:
      #print('adding CE loss as well')
      loss_ce = self.loss_fn(output, targets)
      loss = loss_ce + loss_size + loss_class 
      
    
    self.optimizer.zero_grad()
    loss.backward()   
    self.optimizer.step()
    
    
    #loss1 *= self.mu_p    
    gradient_of_loss_wrt_y = torch.autograd.grad(loss1, self.tau)[0]
    #print(f'gradient_of_loss_wrt_y = {gradient_of_loss_wrt_y}, {self.tau}')

    self.tau = self.tau - self.ags.lr_qr * gradient_of_loss_wrt_y


    self.scores_return  = scores
    
    #q_data2 = scores[int((1-self.train_alpha) * len(scores))]
    if self.local_rank == 0:  
      q_data1 = torch.quantile(scores, q = 1-self.train_alpha).detach().item()
      self.param_taus.append(self.tau.data)
      self.data_taus.append(q_data1)
      
      self.stats['loss_all'].append(loss.item())
      self.stats['loss_pinball'].append(loss1.item())
      self.stats['loss_size'].append(loss_size.item())
      self.stats['class_size_loss'].append(loss_class.item())
      self.stats['CE_loss'].append(loss_ce.item())

      self.stats['scale_soft_set_size'].append(soft_size_loss.item())
      self.stats['indicator_hard_set_size'].append(hard_size_loss.item())
      self.stats['Q_b'].append(q_data1)
      self.stats['Q_param'].append(self.tau.data)
      self.eval_stats['Q_b_train'].append(self.tau.data)
      
    return
    

  def _run_epoch(self, epoch):
    #b_cz = len(next(iter(self.train_data))[0])
    #print(f"GPU: {self.gpu_id} epoch: {epoch} | batch size: {b_cz} | Steps: {len(self.train_data)}")

    loss_all, loss_pinball, loss_size = 0.0, 0.0, 0.0

    for i, (source, targets) in enumerate(self.train_data):
      source = source.to(self.local_rank)
      targets = targets.to(self.local_rank)
      
      if self.method == 'Baseloss': #with CE only
        loss_t = self._run_batch(source, targets)
        loss_all += loss_t.item()
        
      elif self.method == 'pinball_marginal_with_CE':
        loss1, loss2 = self._run_batch_pinball_marginal_with_CE(source, targets)
        loss = loss1 + self.mu_p * loss2
        
      elif self.method == 'only_pinball_marginal': #fine tune with only parameterized pinball loss
        loss_t = self._run_batch_only_pinball_marginal(source, targets)
        loss_all += loss_t.item()
        
      elif self.method == 'pinball_marginal_with_Inefficiency': #fine tune with parameterized pinball loss and inefficiency loss
        self._run_batch_pinball_marginal_with_Inefficiency(source, targets)        
        
      elif self.method == 'pinball_marginal_test': #fine tune with parameterized pinball loss and inefficiency loss
        self._run_batch_pinball_marginal_test(source, targets)   

      elif self.method == 'pinball_marginal_test_2': #fine tune with parameterized pinball loss and inefficiency loss
        self._run_batch_pinball_marginal_test_2(source, targets)  

      elif self.method == 'pinball_class_with_Inefficiency': #fine tune with parameterized class pinball loss and inefficiency loss
        self._run_batch_pinball_class_with_Inefficiency(source, targets)   
        
        
      elif self.method == 'only_Inefficiency': #fine tune with only inefficiency(Deepmind's method)
        self._run_batch_only_Inefficiency(source, targets)  

      elif self.method == 'only_test': #fine tune with only inefficiency(Deepmind's method)
        self._run_batch_only_test(source, targets) 

      elif self.method == 'IW_Inefficiency': #fine tune with only inefficiency(Deepmind's method)
        self._run_batch_IW_Inefficiency(source, targets)   

      elif self.method == 'CPL': 
        if i % 2 == 0:  # Even batch index
          self._run_batch_CPL_optimize_h(source, targets)
        else:  # Odd batch index
          self._run_batch_CPL_optimize_f(source, targets) 

      elif self.method == 'IW_test': #fine tune with only inefficiency(Deepmind's method)
        self._run_batch_IW_test(source, targets)   
      
      elif self.method == 'only_CDF_gap':
        loss_t = self._run_batch_only_CDF_gap(source, targets)
        loss_all += (self.mu*loss_t).item()
                
      elif self.method == 'Conformal':
        loss1, loss2 = self._run_batch_conformal(source, targets)
        loss = loss1 + self.mu * loss2
        
        loss_all += loss.item()
        loss_pinball += loss1.item() #this is basically the NLL loss. 
        loss_size += loss2.item()      #this is basically the CGF gap loss.
                
      elif self.method == 'ConfTr-class':
        loss1, loss2 = self._run_batch_ConfTr_class(source, targets)
        loss = loss1 + loss2
        
      elif self.method == 'ConfTr-marginal':
        loss1, loss2 = self._run_batch_ConfTr_marginal(source, targets)
        loss = loss1 + loss2

      elif self.method == 'pinball-class':
        loss1, loss2, loss3 = self._run_batch_pinball_class(source, targets)
        loss = loss1 + loss2 + loss3 * self.mu_p
        
      elif self.method == 'pinball-marginal':
        loss1, loss2, loss3 = self._run_batch_pinball_marginal_ineff(source, targets)
        loss = loss1 + loss2 + loss3 * self.mu_p
        
      else:
        raise('Specify correct method name')
        
    loss_all = loss_all/len(self.train_data) if len(self.train_data) > 0 else loss_all
    loss_pinball = loss_pinball/len(self.train_data) if len(self.train_data) > 0 else loss_pinball
    loss_size = loss_size/len(self.train_data) if len(self.train_data) > 0 else loss_size
     
    return [loss_all, loss_pinball, loss_size]


    
  def train(self, max_epochs):
    best_acc, best_loss = 0.0, 1e5
        
    if not os.path.exists(self.log_file + '/results.txt'):      
      text_file = open(self.log_file + '/results.txt', 'w')
    else:
      text_file = open(self.log_file + '/results.txt', 'a')


    if self.local_rank == 0:
      if self.method == 'only_Inefficiency' or self.method == 'only_test' or self.method == 'IW_test' or self.method == 'IW_Inefficiency' or self.method == 'CPL' or self.method == 'only_CDF_gap' or self.method == 'pinball_class_with_Inefficiency' or self.method == 'only_pinball_marginal' or ags.method == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2' or self.method == 'pinball_marginal_with_Inefficiency':
        self.param_taus = []
        self.data_taus = []

    self.stats = {              
      'loss_cdf': [], 'CE_loss': [], 'loss_all': [], 'loss_pinball' : [], 'loss_correction' : [], 'loss_addition' : [], 'loss_pinball_over_z' : [], 'lower_loss' : [], 'loss_h' : [], 'loss_lamda' : [], 'lambda' : [], 'loss_size': [], 'Train acc': [], 'Val acc': [], 'num_iter': [], 'class_size_loss': [], 'loss_size_q': [],
      'non_scale_soft_set_size': [], 'scale_soft_set_size': [], 'indicator_hard_set_size': [], 'Q_n': [], 'Q_b': [], 'Q_param': [], 'Z_param': [],'Q_b_diff': [],
    }

    self.eval_stats = {              
      'Q_n_train': [], 'Q_b_train': [], 'Q_n_val': [], 'Q_b_val': [], 'scale_val_soft_set_size': [], 'indicator_val_hard_set_size': [], 
    }


    for e in tqdm(range(self.epochs_run, max_epochs)):

      cleanup_memory()

      if e == self.epochs_run:
        q_hat_quantile = Estimate_quantile_n(model = copy.deepcopy(self.model), dataloader = self.train_data, num_classes = self.num_classes, device = self.local_rank, alpha = self.train_alpha, score_name = self.ags.train_CP_score)
        self.tau = q_hat_quantile.clone().detach().requires_grad_(True).to(self.local_rank)
        self.z = q_hat_quantile.clone().detach().requires_grad_(True).to(self.local_rank)
      
      if e == 0:
        self.qr_decay = 1.0
      else:
        self.qr_decay = math.pow((max(1, e) / self.ags.qr_decay_a), (-self.ags.qr_decay_b))
      
      self.model.train()

      loss = self._run_epoch(e)        
      
      self.scheduler.step()

      
      val_acc = self._run_epoch_val(self.val_data).item()
      train_acc = self._run_epoch_val(self.train_data).item()

      if self.method == 'only_Inefficiency' or self.method == 'only_test' or ags.method == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2' or self.method == 'pinball_marginal_with_Inefficiency' or self.method == 'CPL': 
        self._run_epoch_val_only(self.val_data)

      elif self.method == 'IW_Inefficiency' or self.method == 'IW_test': 
        self._run_epoch_val_IW(self.val_data)

      if self.local_rank == 0 and e + 1 == max_epochs:
        self._save_checkpoint(name = self.save_path +'/final_epoch={}.pt'.format(e+1))
        
      if self.local_rank == 0 and val_acc > best_acc:
        best_acc = val_acc
        self._save_checkpoint(name = self.save_path + '/best_acc.pt')

      if self.local_rank == 0 and loss[0] < best_loss:
        best_loss = loss[0]
        self._save_checkpoint(name = self.save_path + '/best_loss.pt')  
        


        
      if self.local_rank == 0:
        self.stats['Train acc'].append(train_acc)
        self.stats['Val acc'].append(val_acc)
        
        #print(f'Epoch:{e+1} | Train Acc : {train_acc} | Val Acc : {val_acc} | Train loss : {loss}')
        line = f'Epoch:{e+1} | Train Acc : {round(train_acc, 4)} | Val Acc : {round(val_acc, 4)} | Train total loss : {round(loss[0], 4)}\
          | Train pinball loss : {round(loss[1], 4)} | Train Inefficiency loss : {round(loss[2], 4)}\n'
        text_file.write(line)
        
      with torch.no_grad():
        q_hat_quantile = Estimate_quantile_n(model = copy.deepcopy(self.model), dataloader = self.train_data, num_classes = self.num_classes, device = self.local_rank, alpha = self.train_alpha, score_name = self.ags.train_CP_score)
        self.stats['Q_n'].append(q_hat_quantile.item())
        self.eval_stats['Q_n_train'].append(q_hat_quantile.item())

        q_hat_eval = Estimate_quantile_n(model = copy.deepcopy(self.model), dataloader = self.val_data, num_classes = self.num_classes, device = self.local_rank, alpha = self.train_alpha, score_name = self.ags.train_CP_score)
        self.eval_stats['Q_n_val'].append(q_hat_eval.item())
      
      if self.local_rank == 0 and (e+1) % self.save_every == 0:
        self._save_snapshot(e+1, name = self.save_path + '/snapshot_epoch_{}.pt'.format(e+1))  
        
    text_file.close()
    
    if self.local_rank == 0:
      if self.method == 'only_Inefficiency' or self.method == 'CPL' or self.method == 'only_test' or self.method == 'IW_test' or self.method == 'IW_Inefficiency' or self.method == 'only_CDF_gap' or self.method == 'pinball_class_with_Inefficiency' or self.method == 'only_pinball_marginal' or ags.method == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2' or self.method == 'pinball_marginal_with_Inefficiency':

        taus_dict = {}
        taus_dict['param_taus'] = self.param_taus
        taus_dict['method'] = self.method
        taus_dict['data_taus'] = self.data_taus
        taus_dict['scores'] = self.scores_return

        
        torch.save(taus_dict, self.save_path + '/all_taus.pt')   
        
        if self.method == 'only_pinball_marginal' or self.method == 'pinball_marginal_with_Inefficiency1':
          
          save_plot(self.ags, self.scores_return, self.save_path + '/scores_dist.png', self.param_taus, self.data_taus)
        


  def _run_epoch_val(self, data):
    self.model.eval()
    mean_correct = 0.0
    with torch.no_grad():
      for source, targets in data:
        source = source.to(self.local_rank)
        targets = targets.to(self.local_rank)
        pred_targets = torch.argmax(self.model(source), dim = 1)
        mean_correct += torch.mean((targets == pred_targets).float())
        
    #print(f'mean_correct = {len(data)}')
    return mean_correct/len(data) if len(data) > 0 else mean_correct
      

  def _run_epoch_val_only(self, data):
    self.model.eval()
    with torch.no_grad():
      for source, targets in data:
        source = source.to(self.local_rank)
        targets = targets.to(self.local_rank)
        output = self.model(source)
        Temp = self.ags.train_T
        output /= Temp
        ind1, ind2 = train_test_split(torch.arange(len(output)), train_size=0.499, random_state=1111) 
    
        if self.ags.train_CP_score == 'HPS':
          softmax_scores_1 = self.soft_proba(output[ind1])
          softmax_scores_2 = self.soft_proba(output[ind2])
          scores = get_HPS_scores(softmax_scores_1, targets[ind1])
          tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)
          soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_HPS_hard(softmax_scores_2, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         


        elif self.ags.train_CP_score == 'APS':
          softmax_scores_1 = self.soft_proba(output[ind1])
          softmax_scores_2 = self.soft_proba(output[ind2])
          sos = get_sos(self.num_classes)
          scores = smooth_aps_score(softmax_scores_1, targets[ind1], sos=sos, device =self.local_rank, dispersion=0.1, rng=None)
          tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)
          soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_APS_hard(softmax_scores_2, targets[ind2], tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         


        elif self.ags.train_CP_score == 'RAPS':
          scores = compute_scores_diff_RAPS(self.soft_proba(output), targets, device = self.local_rank, lambda_RAPS =.01, k_RAPS =5)
          tau = Smoothquantile(scores, alpha = self.train_alpha, device = self.local_rank)
          soft_size_loss, hard_size_loss, class_loss = Estimate_size_loss_RAPS_hard(scores, targets, self.tau, device = self.local_rank, num_classes = self.num_classes, T = self.T, K = 1.0)         

    
        if self.local_rank == 0:  
          self.eval_stats['Q_b_val'].append(tau)
          self.eval_stats['scale_val_soft_set_size'].append(soft_size_loss)
          self.eval_stats['indicator_val_hard_set_size'].append(hard_size_loss)
          

        
  def _save_checkpoint(self, name):    
    torch.save(self.model.module.state_dict(), name)
    #print(f'save checkpoint at epoch = {name.split}')

  def _save_snapshot(self, epoch, name):
    snapshot = {}
    snapshot['MODEL_STATE'] = self.model.module.state_dict()
    snapshot['EPOCH'] = epoch
    snapshot['stats'] = self.stats
    snapshot['eval_stats'] = self.eval_stats
    
    torch.save(snapshot, name)
    
    #print(f'save checkpoint at epoch = {name.split}')
              


def main(ags, train_sample_dataset, hout_sample_dataset, file_final, log_file):
  
  ddp_setup()
    
  if ags.finetune:
    optimi = ags.finetune_optimizer 
    ler = ags.finetune_lr
    bs = ags.finetune_batch_size
    n_epochs = ags.finetune_epochs
  else:
    optimi = ags.base_optimizer
    ler = ags.base_lr
    bs = ags.batch_size
    n_epochs = ags.num_epochs
    
    
  model, optimizer = load_train_objs(ags, model_name = ags.arc, method_name = ags.method, lr = ler, optimizer_name = optimi)
    
  #print('0011', optimizer.param_groups[0]['params'][-1])

  if ags.finetune:
    path = base_path_for_finetune(ags) #load_train_objs, base_path_for_finetune, load_checkpoint, prepare_dataloader, loss_fnc, check_path, create_final_data, create_folder, test_model
    print(f'path = {path}')
    print('Loading the base model')
    #exit(1)
    model = load_checkpoint(model, path+'/best_acc.pt')

  if ags.method == 'CPL':
    model.fc_single = nn.Linear(model.inplanes, 1).to(model.device)
    nn.init.kaiming_normal_(model.fc_single.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(model.fc_single.bias, 0)

    # Update optimizer to include fc_single parameters
    optimizer.add_param_group({"params": model.fc_single.parameters()})
    
  train_data, val_data = prepare_dataloader(train_sample_dataset, hout_sample_dataset, bs)
  loss_fn = loss_fnc(train_rule = ags.train_rule, cls_num_list = None, num_epochs = ags.num_epochs, loss_type = ags.baseloss)
  cdf_gap = UniformMatchingLoss()

  #print(f'ags = {ags}')
  #exit(1)
  
  if check_path(ags, file_final):
    trainer = Trainer(model, cdf_gap, train_data, val_data, optimizer, loss_fn, mu = ags.mu,\
      mu_size = ags.mu_size, mu_class = ags.mu_class, batch_size = ags.batch_size, save_path = file_final, log_file = log_file, train_alpha = ags.train_alpha,\
      method = ags.method, num_classes = ags.num_classes, save_every = ags.save_every, ags = ags)
  
    trainer.train(n_epochs)    
  
  destroy_process_group()

def main_classwise(ags, model, train_sample_dataset, hout_sample_dataset, file_final, log_file):
  
  ddp_setup()
  assert ags.method == 'pinball_class_with_Inefficiency'
  
    
    
  if ags.finetune:
    optimi = ags.finetune_optimizer 
    ler = ags.finetune_lr
    bs = ags.finetune_batch_size
    n_epochs = ags.classwise_epochs

    
    
  _, optimizer = load_train_objs(ags, model_name = ags.arc, method_name = ags.method, lr = ler, optimizer_name = optimi)
    
    
  train_data, val_data = prepare_dataloader(train_sample_dataset, hout_sample_dataset, bs)
  loss_fn = loss_fnc(train_rule = ags.train_rule, cls_num_list = None, num_epochs = ags.num_epochs, loss_type = ags.baseloss)
  cdf_gap = UniformMatchingLoss()

  trainer = Trainer(model, cdf_gap, train_data, val_data, optimizer, loss_fn, mu = ags.mu, mu_p = ags.mu_p,\
      mu_size = ags.mu_size, mu_class = ags.mu_class, batch_size = ags.batch_size, save_path = file_final, log_file = log_file, train_alpha = ags.train_alpha,\
      method = ags.method, num_classes = ags.num_classes, pinball_loss = pinball, save_every = ags.save_every, ags = ags)
  
  trainer.train(n_epochs)    
  
  destroy_process_group()
  
def main_APS(ags, model, train_sample_dataset, hout_sample_dataset, file_final, log_file):
  
  ddp_setup()
  

  if ags.method == 'pinball_marginal_with_Inefficiency' or ags.method == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2':
    pinball = PinballMarginal()
  else:
    pinball = None

    
  if ags.APS_training and not ags.RAPS_training:
    ags.train_CP_score = 'APS'
    n_epochs = ags.APS_epochs
  if not ags.APS_training and ags.RAPS_training:
    ags.train_CP_score = 'RAPS'
    n_epochs = ags.RAPS_epochs


  optimi = ags.finetune_optimizer 
  ler = ags.finetune_lr
  bs = ags.finetune_batch_size
  

  print(f'ags = {ags}')

    
  _, optimizer = load_train_objs(ags, model_name = ags.arc, method_name = ags.method, lr = ler, optimizer_name = optimi, pinball_model = pinball)
    
    
  train_data, val_data = prepare_dataloader(train_sample_dataset, hout_sample_dataset, bs)
  loss_fn = loss_fnc(train_rule = ags.train_rule, cls_num_list = None, num_epochs = ags.num_epochs, loss_type = ags.baseloss)
  cdf_gap = UniformMatchingLoss()

  trainer = Trainer(model, cdf_gap, train_data, val_data, optimizer, loss_fn, mu = ags.mu, mu_p = ags.mu_p,\
      mu_size = ags.mu_size, mu_class = ags.mu_class, batch_size = ags.batch_size, save_path = file_final, log_file = log_file, train_alpha = ags.train_alpha,\
      method = ags.method, num_classes = ags.num_classes, pinball_loss = pinball, save_every = ags.save_every, ags = ags)
  
  trainer.train(n_epochs)    
  
  destroy_process_group()
  

if __name__ == "__main__":

  parser = argparse.ArgumentParser(add_help=False)

  parser_from_file = parser_file.initialize(parser)

  ags = parser_from_file.parse_args()
  
  #print(ags)
  
  if ags.method == 'Baseloss':
    ags.finetune = False
    ags.mu, ags.mu_p, ags.mu_size = 0.0, 0.0, 0.0
    
  if ags.method == 'Conformal' or ags.method == 'only_CDF_gap':
    ags.CDF_shift = 'KS'
    ags.mu_p, ags.mu_size = 0.0, 0.0
    assert ags.mu != 0.0
    
  if ags.method == 'CPL' or ags.method == 'only_Inefficiency' or ags.method == 'only_test' or ags.method == 'IW_Inefficiency' or ags.method == 'IW_test':
    ags.mu = 0.0
    ags.mu_p = 0.0
    assert ags.mu_size != 0
    
  
  train_sample_dataset, hout_sample_dataset, value = create_final_data(ags, num_classes = ags.num_classes)
  file_final, log_file, file_name = create_folder(ags, value)

  # print(file_final)
    
  main(ags, train_sample_dataset, hout_sample_dataset, file_final, log_file)
    
  if ags.classwise_training:
    print(f'Class wise Training')
    model = test_model(ags, model_name = ags.arc, path = file_final + '/best_acc.pt', previous = False)
    ags.method = 'pinball_class_with_Inefficiency'
    file_final, log_file, file_name = create_folder(ags, value)
    file_final = file_final + '/classwise_epochs={}'.format(ags.classwise_epochs)
    log_file = log_file + '/classwise_epochs={}'.format(ags.classwise_epochs)
    file_name = file_name + '/classwise_epochs={}'.format(ags.classwise_epochs)
    if not os.path.exists(file_final):
      try:
        os.makedirs(file_final)
      except FileExistsError:
        print('File already exists')    

    if not os.path.exists(log_file):
      try:
        os.makedirs(log_file)
      except FileExistsError:
        print('File already exists')  
        
        
    if not os.path.exists(file_final + '/best_acc.pt'):
      main_classwise(ags, model, train_sample_dataset, hout_sample_dataset, file_final, log_file)
    else:
      print('Class wise model exists.')
    

  if ags.finetune:
    if ags.classwise_training:
      epo = ags.classwise_epochs
    else:
      epo = ags.finetune_epochs
      
  if ags.APS_training or ags.RAPS_training:
    print(f'APS Training = {ags.APS_training}, RAPS training = {ags.RAPS_training}')
    
    if ags.method == 'CPL':
      use_fc_single=True
    else:
      use_fc_single=False

    model = test_model(ags, model_name = ags.arc, path = file_final + '/best_acc.pt', previous = False, use_fc_single=use_fc_single)
    
    file_final, log_file, file_name = create_folder(ags, value)
    
    if ags.APS_training:
      file_final = file_final + '/APS_epochs={}'.format(ags.APS_epochs)
      log_file = log_file + '/APS_epochs={}'.format(ags.APS_epochs)
      file_name = file_name + '/APS_epochs={}'.format(ags.APS_epochs)
    
    if ags.RAPS_training:
      file_final = file_final + '/RAPS_epochs={}'.format(ags.APS_epochs)
      log_file = log_file + '/RAPS_epochs={}'.format(ags.APS_epochs)
      file_name = file_name + '/RAPS_epochs={}'.format(ags.APS_epochs)
      
    if not os.path.exists(file_final):
      try:
        os.makedirs(file_final)
      except FileExistsError:
        print('File already exists')    

    if not os.path.exists(log_file):
      try:
        os.makedirs(log_file)
      except FileExistsError:
        print('File already exists')  
        
        
    if not os.path.exists(file_final + '/best_acc.pt'):
      main_APS(ags, model, train_sample_dataset, hout_sample_dataset, file_final, log_file)
    else:
      print('model exists.')
    

  if ags.finetune:
    if ags.classwise_training:
      epo = ags.classwise_epochs
    elif ags.APS_training:
      epo = ags.APS_epochs
    elif ags.RAPS_training:
      epo = ags.RAPS_epochs
    else:
      epo = ags.finetune_epochs
  
  else:
    epo = ags.num_epochs
      
      
  """We can load all three different types of models: fully trained on full epochs, best accurate model, best loss model.
  """
  _,_,test_loader_all,_,_, _, _,_ = \
  load_cifar100(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,num_classes=ags.num_classes)
    
    
  black_boxes_path = ['/final_epoch={}.pt'.format(epo), '/best_acc.pt', '/best_loss.pt']
  
  black_boxes_path_prev = ['/last_model.pt', '/acc', '/loss']

  black_boxes_names = ["bbox","bbox_es_acc","bbox_es_loss"]

  if ags.marginal_calibration:
    save_path1 = './ALLRESULTS/AllResults_1/{}/'.format(value) + '/calibration = {}/CP_score={}/test_alpha={}'.format('MCP', ags.cal_test_CP_score, ags.test_alpha) +'/coverage_on_label={}'.format(ags.coverage_on_label)+ '/num_experiments_' + str(ags.splits) + '/cal_temp={}'.format(ags.cal_test_T) + '/' + file_name

  elif ags.class_wise_calibration:    
    save_path1 = './ALLRESULTS/AllResults_1/{}/'.format(value) + '/calibration = {}/CP_score={}/test_alpha={}'.format('CCP', ags.cal_test_CP_score, ags.test_alpha) +'/coverage_on_label={}'.format(ags.coverage_on_label)+ '/num_experiments_' + str(ags.splits) + '/cal_temp={}'.format(ags.cal_test_T) + '/' + file_name


  RESULTS = pd.DataFrame()
  THRESHOLDS = pd.DataFrame()
  

    
    
  print("begin evaluation.")
  local_rank = int(os.environ["LOCAL_RANK"])

  for i in range(len(black_boxes_path)):
    
    save_path = save_path1 + '/' + black_boxes_names[i] + '/'

    # print(file_final + black_boxes_path_prev[i])

    if not os.path.exists(save_path):
      
      try:
        os.makedirs(save_path)
      except FileExistsError:
        print('File already exists')

    if ags.method == 'CPL':
      use_fc_single=True
    else:
      use_fc_single=False
        
    if ags.previous:
      model = test_model(ags, model_name = ags.arc, path = file_final + black_boxes_path_prev[i], previous = True, use_fc_single=use_fc_single)

    else:
      model = test_model(ags, model_name = ags.arc, path = file_final + black_boxes_path[i], previous = False, use_fc_single=use_fc_single)
    
    
    model.eval()
    if ags.cal_test_CP_score == 'APS':
  
      scores_simple_clean_test, Y_test, y_pred = get_scores(model, test_loader_all, num_classes = ags.num_classes, device=local_rank)
      scores = scores_simple_clean_test[torch.arange(len(scores_simple_clean_test)), Y_test]
      data_frame = pd.DataFrame(
        {'score': scores}
      )
      if local_rank == 0:
        data_frame.to_csv(file_final + '/APS_scores.csv', index = False)

      #exit(1)
    elif ags.cal_test_CP_score == 'HPS':
      scores_simple_clean_test, Y_test, y_pred = get_scores_HPS(model, test_loader_all, num_classes = ags.num_classes, device=local_rank)
      scores = scores_simple_clean_test[torch.arange(len(scores_simple_clean_test)), Y_test]
      data_frame = pd.DataFrame(
        {'score': scores}
      )
      if local_rank == 0:
        data_frame.to_csv(file_final + '/HPS_scores.csv', index = False)

    elif ags.cal_test_CP_score == 'RAPS':
      scores_simple_clean_test, Y_test, y_pred = get_scores_RAPS(model, test_loader_all, num_classes = ags.num_classes, lambda_RAPS =.01, k_RAPS =5, device=local_rank)
      scores = scores_simple_clean_test[torch.arange(len(scores_simple_clean_test)), Y_test]
      data_frame = pd.DataFrame(
        {'score': scores}
      )
      if local_rank == 0:
        data_frame.to_csv(file_final + '/RAPS_scores.csv', index = False)

      #exit(1)
    else:
      raise('Specify correct score')

    results = pd.DataFrame()
    thr = pd.DataFrame()
    
    for experiment in tqdm(range(ags.splits)):
      n_test = len(Y_test)
      idx1, idx2 = train_test_split(np.arange(n_test), train_size=0.499, random_state = experiment + 1111)
    
      if ags.marginal_calibration:
        thresholds, THR, set_matrices = Marginal_conformal(scores_simple_clean_test[idx1, Y_test[idx1]], Y_test[idx1], scores_simple_clean_test[idx2, :], Y_test[idx2], ags.test_alpha,
                          num_classes=ags.num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False)
        
      elif ags.class_wise_calibration:
        thresholds, THR, set_matrices = classwise_conformal(scores_simple_clean_test[idx1, Y_test[idx1]], Y_test[idx1], scores_simple_clean_test[idx2, :], Y_test[idx2], ags.test_alpha,
                          num_classes=ags.num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False)



      THR['Experiment'] = str(experiment + 1)
      thr = pd.concat([thr, THR])

      res = evaluate_predictions(set_matrices, Y_test[idx2], y_pred[idx2], coverage_on_label=ags.coverage_on_label, num_of_classes=ags.num_classes)
              
      res['Experiment'] = str(experiment + 1)

      results = pd.concat([results, res])

    results['method'] = black_boxes_names[i]
    thr['method'] = black_boxes_names[i]
    
    results.to_csv(save_path + '/results_{}.csv'.format(black_boxes_names[i]), index = False)
    thr.to_csv(save_path + '/thresholds_{}.csv'.format(black_boxes_names[i]), index = False)
    
    RESULTS = pd.concat([RESULTS, results])
    THRESHOLDS = pd.concat([THRESHOLDS, thr])

  RESULTS.to_csv(save_path1 + '/results_allMethods.csv', index = False)
  THRESHOLDS.to_csv(save_path1 + '/thresholds_allMethods.csv', index = False)


