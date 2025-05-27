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
import pickle 
import pdb
from conformal_learning.utils import *
from conformal_learning.losses import LDAMLoss, FocalLoss
import argparse
from conformal_learning.pytorch_ops import soft_rank, soft_sort
from conformal_learning.sorting_nets import comm_pattern_batcher
from conformal_learning.variational_sorting_net import VariationalSortingNet
from conformal_learning.smooth_conformal_prediction import smooth_aps_score, smooth_aps_score_all

#import parser_file

#parser = argparse.ArgumentParser(add_help=False)

#parser_from_file = parser_file.initialize(parser)

#args = parser_from_file.parse_args()
import jax
# rng = jax.random.PRNGKey(42) 

import os
os.environ['JAX_NUMPY_DTYPE_PROMOTION'] = 'relaxed'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.3"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc*100

def eval_predictions(test_loader, box, data="unknown", plot=False, predict_1=False):
    if predict_1:
        Y_pred = box.predict_1(test_loader)
    else:
        Y_pred = box.predict(test_loader)
    
    Y_true = []
    for X_batch, Y_batch, _ in test_loader:
      Y_true.append(Y_batch.cpu().numpy()[0])
    
    if plot:
        A = confusion_matrix(Y, Y_pred)
        df_cm = pd.DataFrame(A, index = [i for i in range(K)], columns = [i for i in range(K)])
        plt.figure(figsize = (4,3))
        pal = sns.light_palette("navy", as_cmap=True)
        sn.heatmap(df_cm, cmap=pal, annot=True, fmt='g')

    class_error = np.mean(Y_true!=Y_pred)
    print("Classification error on {:s} data: {:.1f}%".format(data, class_error*100))
    return (class_error*100)

def cvm(u):
  """
  Compute the Cramer von Mises statistic for testing uniformity in distribution
  """
  n = len(u)
  u_sorted = np.sort(u)
  i_seq = (2.0*np.arange(1,1+n)-1.0)/(2.0*n)
  stat = np.sum(np.square(i_seq - u_sorted)) + 1.0/(12.0*n)
  return stat


def KL(P,Q):
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(np.multiply(P,np.log(np.divide(P,Q))),1)
    return divergence


# soft sorting and ranking
REG_STRENGTH = 0.1
B = 50
def soft_indicator(x, a, b=B):
  #print(f'all device = {x.device, a.device}')
  def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
  out = torch.sigmoid(b*(x-a+0.5)) - (torch.sigmoid(b*(x-a-0.5)))
  out = out / (sigmoid(b*(0.5)) - (sigmoid(b*(-0.5))) )
  return out

def soft_indexing(z, rank):
    device = z.device
    n = len(rank)
    K = z.shape[1]
    I = torch.tile(torch.arange(K, device=device), (n,1))
    # Note: this function is vectorized to avoid a loop
    weight = soft_indicator(I.T, rank).T
    z = z.to(device)

    weight = weight * z
    return weight.sum(dim=1)

# The APS non-conformity score
def APS_smooth_scores(probabilities, labels, device, u=None, all_combinations=True):

    # whether to do a randomized score or not
    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = torch.shape(probabilities)[0]

    # sort probabilities from high to low
    #sorted_probabilities = -np.sort(-probabilities)
    sorted_probabilities = -soft_sort(-probabilities, regularization_strength=REG_STRENGTH,device=device)

    # create matrix of cumulative sum of each row
    cumulative_sum = torch.cumsum(sorted_probabilities, axis=1)

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = soft_rank(-probabilities, regularization_strength=REG_STRENGTH,device=device)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = soft_rank(-probabilities, regularization_strength=REG_STRENGTH,device=device)[torch.arange(num_of_points), labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[torch.arange(num_of_points), label_ranks.T].T

    # compute the probability of the last label that enters
    last_label_prob = sorted_probabilities[torch.arange(num_of_points), label_ranks.T].T

    # remove the last label probability or a multiplier of it in the randomized score
    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - torch.diag(u) @ last_label_prob

    # return the scores
    return scores


def compute_scores_diff(proba_values, Y_values, device):
    device = proba_values.device
    """
    Compute the conformity scores and estimate the size of 
    the conformal prediction sets (differentiable) 
    """
    n, K = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
    # Normalize the probabilities again
    proba_values = proba_values / torch.sum(proba_values,1)[:,None]
    # Sorting and ranking
    ranks_array_t = soft_rank(-proba_values, regularization_strength=REG_STRENGTH,device=device)-1
    prob_sort_t = -soft_sort(-proba_values, regularization_strength=REG_STRENGTH,device=device)
    prob_sort_t = prob_sort_t.to(device)
    # Compute the CDF
    Z_t = prob_sort_t.cumsum(dim=1)
    ranks_array_t = ranks_array_t.to(device)
    Y_values = Y_values.to(device)
    # Collect the ranks of the observed labels
    ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(n,1)).flatten()
    # Compute the PDF at the observed labels
    prob_cum_t = soft_indexing(Z_t, ranks_t)
    # Compute the PMF of the observed labels
    prob_final_t = soft_indexing(prob_sort_t, ranks_t)
    # Compute the conformity scores
    scores_t = prob_cum_t - prob_final_t * torch.rand(n,dtype=float,device=device)
    return scores_t


def compute_scores_diff_RAPS(proba_values, Y_values, device, lambda_RAPS = 0.01, k_RAPS = 5.0):
    device = proba_values.device
    """
    Compute the conformity scores and estimate the size of 
    the conformal prediction sets (differentiable) 
    """
    n, K = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
    # Normalize the probabilities again
    proba_values = proba_values / torch.sum(proba_values,1)[:,None]
    # Sorting and ranking
    ranks_array_t = soft_rank(-proba_values, regularization_strength=REG_STRENGTH,device=device)-1
    prob_sort_t = -soft_sort(-proba_values, regularization_strength=REG_STRENGTH,device=device)
    prob_sort_t = prob_sort_t.to(device)
    # Compute the CDF
    Z_t = prob_sort_t.cumsum(dim=1)
    ranks_array_t = ranks_array_t.to(device)
    Y_values = Y_values.to(device)
    # Collect the ranks of the observed labels
    ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(n,1)).flatten()
    # Compute the PDF at the observed labels
    prob_cum_t = soft_indexing(Z_t, ranks_t)
    # Compute the PMF of the observed labels
    prob_final_t = soft_indexing(prob_sort_t, ranks_t)
    # Compute the conformity scores
    scores_t = prob_cum_t - prob_final_t * torch.rand(n,dtype=float,device=device)
    #print(f'scores_t = {scores_t.shape}, {ranks_array_t.shape}, {ranks_t.shape}')
    
    scores_t = scores_t + torch.maximum(lambda_RAPS * (ranks_t.to(device) - k_RAPS), torch.zeros(ranks_t.shape).to(device)) 

    return scores_t

# def get_HPS_scores(softmax_scores, labels):
#     '''
#     Essentially the same as get_APS_scores() except with regularization.
#     See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
#     Inputs:
#         softmax_scores: n x num_classes
#         labels: length-n array of class labels
#         lmbda, kreg: regularization parameters
#     Output: 
#         length-n array of APS scores
    
#     '''
#     true_scores = softmax_scores[np.arange(len(labels)), labels]
    
#     scores = np.array(1 - true_scores)
#     # scores = 1 - true_scores
    
#     return scores

def get_HPS_scores(softmax_scores, labels):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes (tensor)
        labels: Length-n tensor of class labels
    
    Output: 
        Length-n tensor of APS scores
    '''
    # Extract true label scores for each sample
    true_scores = softmax_scores[torch.arange(len(labels)), labels]
    
    # Compute conformity scores as 1 - true_scores
    scores = 1 - true_scores
    
    return scores

def get_HPS_scores_all(softmax_scores):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
        lmbda, kreg: regularization parameters
    Output: 
        length-n array of APS scores
    
    '''
    scores_all = 1 - softmax_scores
    
    # scores = np.array(scores_all)

    return scores
    

# def get_APS_scores(softmax_scores, labels, randomize=True, seed=0):
#     '''
#     Compute conformity score defined in Romano et al, 2020
#     (Including randomization, unless randomize is set to False)
    
#     Inputs:
#         softmax_scores: n x num_classes
#         labels: length-n array of class labels
    
#     Output: 
#         length-n array of APS scores
#     '''
#     n = len(softmax_scores)
#     sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
#     scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
#     scores = np.array(scores)
    
#     if not randomize:
#         return scores - softmax_scores[range(n), labels]
#     else:
#         np.random.seed(seed)
#         U = np.random.rand(n) # Generate U's ~ Unif([0,1])
#         randomized_scores = scores - U * softmax_scores[range(n), labels]
#         return randomized_scores

def get_APS_scores(softmax_scores, labels, randomize=True, seed=0):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: PyTorch tensor of shape (n, num_classes)
        labels: Length-n tensor of class labels
    
    Output: 
        Length-n tensor of APS scores
    '''
    n = len(softmax_scores)
    
    # Sort softmax scores in descending order and get sorted indices
    sorted_scores, pi = softmax_scores.sort(dim=1, descending=True)  # pi is the indices in the original array

    # Cumulative sum of sorted scores and gather back in the original order
    scores = sorted_scores.cumsum(dim=1).gather(1, pi.argsort(1))[torch.arange(n), labels]

    if not randomize:
        return scores - softmax_scores[torch.arange(n), labels]
    else:
        # Set seed and generate uniform random numbers with PyTorch
        torch.manual_seed(seed)
        U = torch.rand(n, device=softmax_scores.device)  # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[torch.arange(n), labels]
        return randomized_scores

# def get_APS_scores_all(softmax_scores, randomize=True, seed=0):
#     '''
#     Similar to get_APS_scores(), except the APS scores are computed for all 
#     classes instead of just the true label
    
#     Inputs:
#         softmax_scores: n x num_classes
    
#     Output: 
#         n x num_classes array of APS scores
#     '''
#     n = softmax_scores.shape[0]
#     sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
#     scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
#     scores = np.array(scores)
    
#     if not randomize:
#         return scores - softmax_scores
#     else:
#         np.random.seed(seed)
#         U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
#         randomized_scores = scores - U * softmax_scores # [range(n), labels]
#         return randomized_scores

def get_APS_scores_all(softmax_scores, randomize=True, seed=0):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label.
    
    Inputs:
        softmax_scores: PyTorch tensor of shape (n, num_classes)
    
    Output: 
        PyTorch tensor of shape (n, num_classes) with APS scores
    '''
    n = softmax_scores.shape[0]
    
    sorted_scores, pi = softmax_scores.sort(dim=1, descending=True)  # pi is the indices in the original array

    # Cumulative sum of sorted scores and reorder to original indices
    scores = sorted_scores.cumsum(dim=1).gather(1, pi.argsort(dim=1))

    if not randomize:
        return scores - softmax_scores
    else:
        # Set seed and generate uniform random numbers in PyTorch
        torch.manual_seed(seed)
        U = torch.rand_like(softmax_scores, device=softmax_scores.device)  # Generate U ~ Unif([0,1]) with same shape as softmax_scores
        randomized_scores = scores - U * softmax_scores
        return randomized_scores

def get_RAPS_scores(softmax_scores, labels, lmbda=0.01, kreg=5, randomize=True, seed=0):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes (tensor)
        labels: Length-n tensor of class labels
        lmbda, kreg: regularization parameters
    
    Output: 
        Length-n tensor of APS scores
    '''
    n = len(softmax_scores)
    
    # Sort softmax scores in descending order and get indices
    sorted_scores, pi = softmax_scores.sort(dim=1, descending=True)
    
    # Cumulative sum and gather in original order
    scores = sorted_scores.cumsum(dim=1).gather(1, pi.argsort(dim=1))[torch.arange(n), labels]
    
    # Regularization
    y_rank = pi.argsort(dim=1)[torch.arange(n), labels] + 1  # Rank of the true labels
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros_like(y_rank))
    scores += reg

    if not randomize:
        return scores - softmax_scores[torch.arange(n), labels]
    else:
        torch.manual_seed(seed)
        U = torch.rand(n, device=softmax_scores.device)  # Uniform random numbers
        randomized_scores = scores - U * softmax_scores[torch.arange(n), labels]
        return randomized_scores


def get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True, seed=0):
    '''
    Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
    classes instead of just the true label.
    
    Inputs:
        softmax_scores: n x num_classes (tensor)
    
    Output: 
        n x num_classes tensor of APS scores
    '''
    n = softmax_scores.shape[0]
    
    # Sort softmax scores in descending order and get indices
    sorted_scores, pi = softmax_scores.sort(dim=1, descending=True)
    
    # Cumulative sum and reorder
    scores = sorted_scores.cumsum(dim=1).gather(1, pi.argsort(dim=1))

    # Regularization for each class as if it were the true label
    y_rank = pi.argsort(dim=1) + 1  # Rank for each class
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros_like(scores))
    scores += reg

    if not randomize:
        return scores - softmax_scores
    else:
        torch.manual_seed(seed)
        U = torch.rand_like(softmax_scores)  # Uniform random numbers
        randomized_scores = scores - U * softmax_scores
        return randomized_scores

# def get_RAPS_scores(softmax_scores, labels, lmbda=.01, kreg=5, randomize=True, seed=0):
#     '''
#     Essentially the same as get_APS_scores() except with regularization.
#     See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
#     Inputs:
#         softmax_scores: n x num_classes
#         labels: length-n array of class labels
#         lmbda, kreg: regularization parameters
#     Output: 
#         length-n array of APS scores
    
#     '''
#     n = len(labels)
#     sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
#     scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    
#     # Regularization
#     y_rank = pi.argsort(1)[range(labels_calib.shape[0]), labels_calib] + 1 # Compute softmax rank of true labels y
#     reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=y_rank.shape))
#     scores += reg
    
#     scores = np.array(scores)
    
#     if not randomize:
#         return scores - softmax_scores[range(n), labels]
#     else:
#         np.random.seed(seed)
#         U = np.random.rand(n) # Generate U's ~ Unif([0,1])
#         randomized_scores = scores - U * softmax_scores[range(n), labels]
#         return randomized_scores


# def get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True, seed=0):
#     '''
#     Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
#     classes instead of just the true label
    
#     Inputs:
#         softmax_scores: n x num_classes
    
#     Output: 
#         n x num_classes array of APS scores
#     '''
#     n = softmax_scores.shape[0]
#     sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
#     scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    
#     # Regularization (pretend each class is true label)
#     y_rank = pi.argsort(1) + 1 # Compute softmax rank of true labels y
#     reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=scores.shape))
 
#     scores += reg
        
#     if not randomize:
#         return scores - softmax_scores
#     else:
#         np.random.seed(seed)
#         U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
#         randomized_scores = scores - U * softmax_scores # [range(n), labels]
#         return randomized_scores

    
    
# def get_HPS_scores_all(softmax_scores):
#     return 1 - softmax_scores
    
def get_conformal_quantile(scores:np.ndarray, alpha:float, default_qhat:np.inf, exact_coverage:False):
    '''
    Compute finite-sample-adjusted 1-alpha quantile of scores
    
    Inputs:
        - scores: num_instances-length array of conformal scores for true class. A higher score 
            indicates more uncertainty
        - alpha: float between 0 and 1 specifying coverage level
        - default_qhat: the value that will be returned if there are insufficient samples to compute
        the quantile. Should be np.inf if you want a coverage guarantee.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    
    '''
    if exact_coverage:
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        return exact_cov_params
    
    else:
        n = len(scores)
        if default_qhat == 'max':
            default_qhat = np.max(scores)

        if n == 0:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            return default_qhat

        val = np.ceil((n+1)*(1-alpha))/n
        if val > 1:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            qhat = default_qhat
        else:
            qhat = np.quantile(scores, val)

        return qhat

def get_conformal_quantile_2(scores:np.ndarray, alpha:float, default_qhat:np.inf, exact_coverage:False):
    '''
    Compute finite-sample-adjusted 1-alpha quantile of scores
    
    Inputs:
        - scores: num_instances-length array of conformal scores for true class. A higher score 
            indicates more uncertainty
        - alpha: float between 0 and 1 specifying coverage level
        - default_qhat: the value that will be returned if there are insufficient samples to compute
        the quantile. Should be np.inf if you want a coverage guarantee.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    
    '''
    if exact_coverage:
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        return exact_cov_params
    
    else:
        n = len(scores)
        if default_qhat == 'max':
            default_qhat = np.max(scores)

        if n == 0:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            return default_qhat

        val = (np.ceil((n+1)*(1-alpha)) - 1)/n
        if val > 1:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            qhat = default_qhat
        else:
            qhat = np.quantile(scores, val)

        return qhat

# def get_conformal_quantile(scores: torch.Tensor, alpha: float, default_qhat: float = float('inf'), exact_coverage: bool = False):
#     '''
#     Compute finite-sample-adjusted 1-alpha quantile of scores
    
#     Inputs:
#         - scores: Tensor of conformal scores for the true class. A higher score 
#             indicates more uncertainty
#         - alpha: float between 0 and 1 specifying coverage level
#         - default_qhat: the value that will be returned if there are insufficient samples to compute
#         the quantile. Should be float('inf') if you want a coverage guarantee.
#         - exact_coverage: If True, returns a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
#           such that if you use q_hat = q_a with probability gamma and q_b with probability 1 - gamma,
#           you achieve exact 1-alpha coverage
#     '''
#     if exact_coverage:
#         # Ensure exact coverage parameters use tensor-compatible functions
#         q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=float('inf'))
#         exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
#         return exact_cov_params
    
#     else:
#         n = len(scores)
#         if default_qhat == 'max':
#             default_qhat = torch.max(scores)
        
#         if n == 0:
#             print(f'Using default q_hat of {default_qhat} because n={n}')
#             return default_qhat

#         # Calculate the quantile threshold value
#         val = torch.ceil((n + 1) * (1 - alpha)) / n
#         if val > 1:
#             print(f'Using default q_hat of {default_qhat} because n={n}')
#             qhat = default_qhat
#         else:
#             qhat = torch.quantile(scores, val)

#         return qhat

def estimate_class_quantiles(num_classes:int, cal_scores_all:np.ndarray, cal_true_labels:np.ndarray, alpha:float, default_qhat:np.inf):

  class_cts = np.zeros((num_classes,))
  q_hats = np.zeros((num_classes,)) #q_hats[i] = quantile for class i

  for k in range(num_classes):

      # Only select data for which k is true class
      idx = (cal_true_labels == k).detach().cpu().numpy()
      scores = cal_scores_all[idx]

      class_cts[k] = scores.shape[0]

      q_hats[k] = get_conformal_quantile(scores=scores, alpha=alpha, default_qhat=default_qhat,exact_coverage = False)
      
  q_hats = torch.from_numpy(q_hats)
  #print(f"q_hats = {q_hats}")
  return q_hats


def compute_HPS_scores(train_proba, Y_batch, device):
    #print(f"train_proba = {train_proba.shape}")
    #print(f"{Y_batch.shape}")
    #print(f"{Y_batch}")
    return 1-train_proba[torch.arange(len(train_proba)), Y_batch]


# def Smoothquantile(scores, alpha, device):
#     n = len(scores)
#     # scores_np = scores.cpu().numpy()
#     scores_np = scores.detach().cpu().numpy()
#     # print(1 - alpha)

#     # Compute q_hat using the provided function with the NumPy array
#     q_hat = get_conformal_quantile(scores=scores_np, alpha=alpha, default_qhat=np.inf, exact_coverage=False)

#     # print(q_hat)

#     # Convert q_hat to a tensor and move it to the specified device
#     q_hat_tensor = torch.tensor(q_hat, device=device, dtype=scores.dtype)
    
#     return q_hat_tensor

def Smoothquantile(scores, alpha, device):
    n = len(scores)
    sorted_score = soft_sort(scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
    #print(f"sorted_score = {sorted_score}, {sorted_score.shape}")
    index = int(n*(1.0-alpha))
    scores_q_t = sorted_score[index]
    #print(f"scores_q_t = {scores_q_t}")
    return scores_q_t

# def Smoothquantile_2(scores, size, alpha, device):
#     n = len(scores)
#     # scores_np = scores.cpu().numpy()
#     scores_np = scores.detach().cpu().numpy()
#     # print(1 - alpha)
#     alpha_plus = np.ceil((n+1)*(1.0-alpha))
#     alpha_minus = np.ceil((n+1)*(1.0-alpha))-1
#     alpha_standard = np.ceil((size+1)*(1.0-alpha))
#     weight = (alpha_standard/size - alpha_minus/(n+1))/(alpha_plus/(n+1) - alpha_minus/(n+1))
#     # Compute q_hat using the provided function with the NumPy array
#     q_hat_plus = get_conformal_quantile(scores=scores_np, alpha=alpha, default_qhat=np.inf, exact_coverage=False)
#     q_hat_minus = get_conformal_quantile_2(scores=scores_np, alpha=alpha, default_qhat=np.inf, exact_coverage=False)

#     q_hat = weight* q_hat_plus + (1-weight)* q_hat_minus
#     # print(q_hat)
#     # Convert q_hat to a tensor and move it to the specified device
#     q_hat_tensor = torch.tensor(q_hat, device=device, dtype=scores.dtype)
    
#     return q_hat_tensor

def Smoothquantile_2(scores, size, alpha, device):
    n = len(scores)
    sorted_score = soft_sort(scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
    #print(f"sorted_score = {sorted_score}, {sorted_score.shape}")
    alpha_plus = int((n+1)*(1.0-alpha))
    alpha_minus = int((n+1)*(1.0-alpha)-1)
    alpha_standard = np.ceil((size+1)*(1.0-alpha))
    weight = (alpha_standard/size - alpha_minus/(n+1))/(alpha_plus/(n+1) - alpha_minus/(n+1))
    scores_q_1 = sorted_score[int(alpha_plus-1)]
    scores_q_2 = sorted_score[int(alpha_minus-1)]
    scores_q_t = weight* scores_q_1 + (1-weight)* scores_q_2
    #print(f"scores_q_t = {scores_q_t}")
    return scores_q_t

def Smoothquantile_3(scores, size, alpha, device):
    n = len(scores)
    sorted_score = soft_sort(scores.reshape((1, n)), regularization_strength=REG_STRENGTH, device = device).flatten()
    #print(f"sorted_score = {sorted_score}, {sorted_score.shape}")
    alpha_plus = int((n+1)*(1.0-alpha))
    alpha_minus = int((n+1)*(1.0-alpha)-1)
    alpha_standard = np.ceil((size+1)*(1.0-alpha))
    weight = (alpha_standard/size - alpha_minus/n)/(alpha_plus/n - alpha_minus/n)
    scores_q_1 = sorted_score[int(alpha_plus-1)]
    scores_q_2 = sorted_score[int(alpha_minus-1)]
    scores_q_t = weight* scores_q_1 + (1-weight)* scores_q_2
    #print(f"scores_q_t = {scores_q_t}")
    return scores_q_t

def Smoothquantile_per_class(device:int, scores:torch.tensor, Y_true:torch.tensor, alpha:float=0.1, num_class:int=100):

    Qs = torch.zeros(size=(num_class, ))

    for i in range(num_class):
        idx_i = torch.where(Y_true == i)[0]
        if len(idx_i) == 0:
            Qs[i] = 1 - alpha
        else:
            scores_i = scores[idx_i]
            sorted_score_i = soft_sort(scores_i.reshape((1, len(scores_i))), regularization_strength=REG_STRENGTH, device=device).flatten()

            Qs[i] = sorted_score_i[int(len(scores_i)*(1.0-alpha))]

    return Qs

############ Clustered CP #########################
def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

def get_rare_classes(labels, alpha, num_classes):
    thresh = get_quantile_threshold(alpha)
    classes, cts = np.unique(labels, return_counts=True)
    rare_classes = classes[cts < thresh]
    
    # Also included any classes that are so rare that we have 0 labels for it
    zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
    rare_classes = np.concatenate((rare_classes, zero_ct_classes))
    
    return rare_classes


def remap_classes(labels, rare_classes):
    '''
    Exclude classes in rare_classes and remap remaining classes to be 0-indexed

    Outputs:
        - remaining_idx: Boolean array the same length as labels. Entry i is True
        iff labels[i] is not in rare_classes 
        - remapped_labels: Array that only contains the entries of labels that are 
        not in rare_classes (in order) 
        - remapping: Dict mapping old class index to new class index

    '''
    remaining_idx = ~np.isin(labels, rare_classes)

    remaining_labels = labels[remaining_idx]
    remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
    new_idx = 0
    remapping = {}
    for i in range(len(remaining_labels)):
        if remaining_labels[i] in remapping:
            remapped_labels[i] = remapping[remaining_labels[i]]
        else:
            remapped_labels[i] = new_idx
            remapping[remaining_labels[i]] = new_idx
            new_idx += 1
    return remaining_idx, remapped_labels, remapping


def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    '''
    return np.quantile(samples, q)


def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Input:
        - scores_all: num_instances x num_classes array where 
            scores_all[i,j] = score of class j for instance i
          Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
        - labels: num_instances-length array of true class labels
        - q: quantiles to include in embedding
        - return_cts: if True, return an array containing the counts for each class 
        
    Output: 
        - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
        - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
        appears in labels 
    '''
    num_classes = len(np.unique(labels))
    
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    
    for i in range(num_classes):
        if len(scores_all.shape) == 2:
            class_i_scores = scores_all[labels==i,i]
        else:
            class_i_scores = scores_all[labels==i] 
        cts[i] = class_i_scores.shape[0]
        embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
    
    if return_cts:
        return embeddings, cts
    else:
        return embeddings
    

def compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_true_labels, alpha, 
                                   null_qhat='standard', exact_coverage=False):
    '''
    Computes cluster-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
          Clusters should be 0-indexed. Rare classes can be assigned to cluster -1 and they will automatically be given
          qhat_k = default_qhat. 
        - cal_scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i.
         Alternatively, a num_instances-length array of conformal scores for true class
        - cal_true_labels: num_instances length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - null_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to null_qhat.
        If null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        - exact_coverage: If True, return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
         
    Output:
        num_classes length array where entry i is the quantile corresponding to the cluster that class i belongs to. 
        All classes in the same cluster have the same quantile.
        
        OR (if exact_coverage=True), dict containing clustered conformal parameters needed to achieve exact coverage
    '''
    # If we want the null_qhat to be the standard qhat, we should compute this using the original class labels
    if null_qhat == 'standard' and not exact_coverage:
        null_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha)
            
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
        
    # Edge case: all cluster_assignments are -1. 
    if np.all(cluster_assignments==-1):
        if exact_coverage:
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conformal params to null cluster
            q_as = null_qa * np.ones(cluster_assignments.shape)
            q_bs = null_qb * np.ones(cluster_assignments.shape)
            gammas = null_gamma * np.ones(cluster_assignments.shape)
            return q_as, q_bs, gammas
        else:
            return null_qhat * np.ones(cluster_assignments.shape)
    
    # Map true class labels to clusters
    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
    
    # Compute cluster qhats
    if exact_coverage:
        if null_qhat == 'standard':
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conforml params to null cluster
            null_params = {'q_a': null_qa, 'q_b': null_qb, 'gamma': null_gamma}
        clustq_as, clustq_bs, clustgammas = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_clusters,
                                                                          num_classes=np.max(cluster_assignments)+1, 
                                                                          alpha=alpha, 
                                                                          default_qhat=np.inf, null_params=null_params)
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        q_as = np.array([clustq_as[cluster_assignments[k]] for k in range(num_classes)])
        q_bs = np.array([clustq_bs[cluster_assignments[k]] for k in range(num_classes)])
        gammas = np.array([clustgammas[cluster_assignments[k]] for k in range(num_classes)])

        return q_as, q_bs, gammas
   
    else:
            
        cluster_qhats = compute_class_specific_qhats(cal_scores_all, cal_true_clusters, 
                                                     alpha=alpha, num_classes=np.max(cluster_assignments)+1,
                                                     default_qhat=np.inf,
                                                     null_qhat=null_qhat)                            
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])

        return class_qhats
    



def get_logits_targets1(model, loader):
    layer_prob = nn.Softmax(dim=1)
    probas = torch.zeros((len(loader.dataset), 100)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),), dtype = torch.int64)
    i = 0
    #print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets, _ in loader:
            batch_logits = model(x.cuda()).detach().cpu()
            proba = layer_prob(batch_logits)
            probas[i:(i+x.shape[0]), :] = proba
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    #dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return probas, labels

def EstimateEntireDataQ(loader, model, alpha):
    probas, labels = get_logits_targets1(model, loader)
    #print(f'shape = {probas.shape} , {labels.shape}')
    scores = compute_HPS_scores(probas, labels)
    return torch.quantile(scores, q=1-alpha)

def soft_union_size_single(proba:torch.tensor, transformer:torch.tensor, theta:torch.tensor, eta:int=10)->float:
    device = proba.device
    probs = 1*proba
    probs = probs + 1e-6*torch.rand(probs.shape, dtype=float, device=device)
    probs = probs / torch.sum(probs, 1)[:,None]
    assert(len(probs.shape) == 2)
    n, K = probs.shape
    assert(n == 1)

    u = torch.rand(K, dtype=float, device=device)

    # ranks
    ranks = soft_rank(-probs, regularization_strength=REG_STRENGTH) - 1
    # sorted probs from highest to lowest
    probs_s = -soft_sort(-probs, regularization_strength=REG_STRENGTH)
    # cumulative sum
    probs_c = probs_s.cumsum(dim=1)
    ranks = ranks.to(device)

    # re-index to original ordering of classes
    probs_c_r = soft_indexing(probs_c, ranks)

    # calculate scores
    scores = probs_c_r - u * probs

    # calculate higher than thresholds
    #print(f"shape = {scores, transformer, theta}")
    diff = torch.matmul(scores, transformer.double()) - theta
    #print(f"torch.matmul(scores, transformer.double()) = {torch.matmul(scores, transformer.double())}, theta = {theta}")

    # soft counting of set sizes
    #print(f"diff = {diff, eta}")
    sgm = torch.sigmoid(diff * eta)
    soft_sizes = sgm.mean(dim=1)
    #print(f"soft_sizes = {soft_sizes}")
    return soft_sizes

def Estimate_size_loss_all_class_APS(proba:torch.tensor, Y_true:torch.tensor, transformer:torch.tensor, theta:torch.tensor, eta:int=0.001, num_classes:int = 100)->tuple:
    n, k = proba.shape
    sizes = 0
    for i in range(n):
        sizes += soft_union_size_single(proba = proba[i,:].view((1, k)), transformer=transformer, theta=theta, eta=eta).sum()

    one_hot = torch.nn.functional.one_hot(Y_true, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba) +  not_one_hot * proba
    class_loss = torch.mean(torch.sum(class_set, dim = 1))

    return (sizes / n, class_loss)


def Estimate_size_loss_APS_Hooman(proba, Y_batch, tau, device, num_classes = 100, T = 1.0, K = 0.0):
    n = len(proba)

    device = proba.device
    probs = 1*proba
    probs = probs + 1e-6*torch.rand(probs.shape, dtype=float, device=device)
    probs = probs / torch.sum(probs, 1, keepdim = True)

    u = torch.rand(K, dtype=float, device=device)

    # ranks
    ranks = soft_rank(-probs, regularization_strength=REG_STRENGTH) - 1
    # sorted probs from highest to lowest
    probs_s = -soft_sort(-probs, regularization_strength=REG_STRENGTH)
    # cumulative sum
    probs_c = probs_s.cumsum(dim=1)
    ranks = ranks.to(device)

    # re-index to original ordering of classes
    probs_c_r = soft_indexing(probs_c, ranks)

    # calculate scores
    scores = probs_c_r - u * probs

    set_preds = torch.sigmoid((scores - tau.to(device))/T)
    
    
    size_loss = torch.mean(torch.maximum(torch.sum(set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    one_hot = torch.nn.functional.one_hot(Y_batch, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba) +  not_one_hot * proba
    class_loss = torch.mean(torch.sum(class_set, dim = 1))
    return size_loss, class_loss

def soft_indexing_all_labels(z, rank):
    device = z.device
    n = len(rank)
    K = z.shape[1]
    I = torch.tile(torch.arange(K, device=device), (n,1))
    I = I.unsqueeze(1).repeat(1, K, 1)
    I = I.to(device)
    rank = rank.unsqueeze(2)
    weight = soft_indicator(I, rank)
    z = z.unsqueeze(1).repeat(1, K, 1)
    weight = weight * z
    return weight.sum(dim=-1)

def estimate_qn(scores, alpha = 0.1):
  n = len(scores)
  scores = np.sort(scores) # ascending order sorting
  ind = int(np.ceil((1 - alpha)*(1 + n)))

  return scores[ind]


def estimate_cardinality1(scores_, qn, T = 0.1):
    """
    scores: (n, ) 
    scores_: (n, K)
    """

    #scores = 1 - f(x, y) or S_APS
    #qn = estimate_qn(scores, alpha = alpha)
    
    soft_set1 = 1 - torch.sigmoid((scores_ - qn)/1) 
    set_size1 = soft_set1.sum(axis = 1).mean()

    soft_set1 = 1 - torch.sigmoid((scores_ - qn)/T) 
    set_size2 = soft_set1.sum(axis = 1).mean()

    soft_set1 = (scores_ <= qn).to(torch.float32)
    set_size3 = soft_set1.sum(axis = 1).mean()

    return set_size1, set_size2, set_size3


def Estimate_size_loss_HPS(proba, Y_batch, tau, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):
    n = len(proba)

    #set_size = torch.mean(torch.maximum(torch.sum(set_preds, dim=1) - target_size, torch.zeros(n).to(device)))
    #print(f"tau = {tau}")
    #print(f'device = {proba.get_device()}, {tau.get_device()}, {device}')

    score = 1 - proba

    set_preds = torch.sigmoid((tau.to(device)- score)/T)
    
    #print(f'T hps = {T}')
    #exit(1)
    if training_size:
        return torch.sum(set_preds, dim = 1) 
    
        #return torch.sum(set_preds, dim = 1) - K
    
    size_loss = torch.mean(torch.maximum(torch.sum(set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    one_hot = torch.nn.functional.one_hot(Y_batch, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba) +  not_one_hot * proba
    class_loss = torch.mean(torch.sum(class_set, dim = 1))


    return size_loss, class_loss


def Estimate_size_loss_APS(proba_values, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):
    """
    Compute the conformity scores and estimate the size of 
    the conformal prediction sets (differentiable) 
    """
    device = proba_values.device
    n, num_classes = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
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
    prob_final_t = soft_indexing_all_labels(prob_sort_t, ranks_array_t)
    # Compute the conformity scores
    uu = torch.rand((n, num_classes) ,dtype=float,device=device)


    score_all_classes = prob_cum_t - prob_final_t * uu
    
    #print(f'score_all_classes = {score_all_classes}')
    

    set_preds = torch.sigmoid((score_all_classes - tau.to(device))/T)
    
    #print(f'T aps = {T}')
    #exit(1)

    if training_size:
        return torch.sum(set_preds, dim = 1) #torch.mean((score_all_classes - tau.to(device))/T).item()
    
    size_loss = torch.mean(torch.maximum(torch.sum(set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    #print(f'size_loss = {size_loss}, set_preds = {set_preds}')
    #exit()

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba_values) +  not_one_hot * proba_values
    class_loss = torch.mean(torch.sum(class_set, dim = 1))
    return size_loss, class_loss

def Estimate_size_loss_RAPS(proba_values, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, lambda_RAPS = 0.01, k_RAPS = 5):
    """
    Compute the conformity scores and estimate the size of
    the conformal prediction sets (differentiable)
    """
    device = proba_values.device
    n, K = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
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
    prob_final_t = soft_indexing_all_labels(prob_sort_t, ranks_array_t)
    # Compute the conformity scores
    uu = torch.rand((n, K) ,dtype=float,device=device)


    """score_all_classes = prob_cum_t + prob_final_t * uu #Differentiable APS score for all classes"""

    score_all_classes = prob_cum_t + prob_final_t * uu + torch.maximum(lambda_RAPS * (ranks_array_t - k_RAPS), torch.zeros(ranks_array_t.shape).to(device)) #Differentiable RAPS score for all classes


    set_preds = torch.sigmoid((score_all_classes - tau.to(device))/T)

    size_loss = torch.mean(torch.maximum(torch.sum(set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba_values) +  not_one_hot * proba_values
    class_loss = torch.mean(torch.sum(class_set, dim = 1))
    return size_loss, class_loss


def Estimate_size_loss_HPS_hard(proba, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):
    n = len(proba)

    #set_size = torch.mean(torch.maximum(torch.sum(set_preds, dim=1) - target_size, torch.zeros(n).to(device)))
    #print(f"tau = {tau}")
    #print(f'device = {proba.get_device()}, {tau.get_device()}, {device}')

    score = 1 - proba

    soft_set_preds = torch.sigmoid((tau.to(device) - score)/T)
    # hard_set_preds = (proba <= tau.to(device)).to(torch.float32)
    
    #print(f'T hps = {T}')
    #exit(1)
    if training_size:
        return torch.sum(set_preds, dim = 1) 
    
        #return torch.sum(set_preds, dim = 1) - K
    
    # soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))
    # hard_size_loss = torch.mean(torch.sum(hard_set_preds, dim = 1))

    # one_hot = torch.nn.functional.one_hot(Y_batch, num_classes = num_classes)
    # not_one_hot = 1 - one_hot
    # class_set = one_hot * (1 - proba) +  not_one_hot * proba
    # class_loss = torch.mean(torch.sum(class_set, dim = 1))

    soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    hard_set1 = (score <= tau).to(torch.float32)
    hard_size_loss = hard_set1.sum(axis = 1).mean()

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - soft_set_preds) +  not_one_hot * soft_set_preds
    class_loss = torch.mean(torch.sum(class_set, dim = 1))


    return soft_size_loss, hard_size_loss, class_loss 


# def Estimate_size_loss_APS_hard(proba_values, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):
#     """
#     Compute the conformity scores and estimate the size of 
#     the conformal prediction sets (differentiable) 
#     """
#     device = proba_values.device
#     n, num_classes = proba_values.shape
#     # Break possible ties at random (it helps with the soft sorting)
#     proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
#     # Normalize the probabilities again
#     proba_values = proba_values / torch.sum(proba_values,1)[:,None]

#     # Sorting and ranking
#     ranks_array_t = soft_rank(-proba_values, device = device, regularization_strength=REG_STRENGTH)-1
#     prob_sort_t = -soft_sort(-proba_values, device = device, regularization_strength=REG_STRENGTH)
    
#     ranks_array_t = ranks_array_t.to(device)
#     prob_sort_t = prob_sort_t.to(device)
    
#     # Compute the CDF
#     Z_t = prob_sort_t.cumsum(dim=1)

#     prob_cum_t = soft_indexing_all_labels(Z_t, ranks_array_t)
#     # Compute the PMF of the observed labels
#     prob_final_t = soft_indexing_all_labels(prob_sort_t, ranks_array_t)
#     # Compute the conformity scores
#     uu = torch.rand((n, num_classes) ,dtype=float,device=device)


#     score_all_classes = prob_cum_t - prob_final_t * uu
    
#     #print(f'score_all_classes = {score_all_classes}')
    

#     soft_set_preds = torch.sigmoid((tau.to(device) - score_all_classes)/T)
#     hard_set_preds = (score_all_classes <= tau.to(device)).to(torch.float32)
    
    
#     #print(f'T aps = {T}')
#     #exit(1)

#     if training_size:
#         return torch.sum(set_preds, dim = 1) #torch.mean((score_all_classes - tau.to(device))/T).item()
    
#     soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))
#     hard_size_loss = torch.mean(torch.sum(hard_set_preds, dim = 1))

#     one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
#     not_one_hot = 1 - one_hot
#     class_set = one_hot * (1 - proba_values) +  not_one_hot * proba_values
#     class_loss = torch.mean(torch.sum(class_set, dim = 1))

#     #print(f'size_loss = {size_loss}, set_preds = {set_preds}')
#     #exit()
#     return soft_size_loss, hard_size_loss, class_loss 

def get_sos(length: int) -> VariationalSortingNet:
    """Set up smooth order stat object for given array length.

    Args:
      length: length of array to be sorted

    Returns:
      Smooth order stat object
    """
    comm = comm_pattern_batcher(length, make_parallel=True)
    sos = VariationalSortingNet(comm, smoothing_strategy='entropy_reg', sorting_strategy='hard')
    return sos

def Estimate_size_loss_APS_hard(proba_values, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):
    """
    Compute the conformity scores and estimate the size of 
    the conformal prediction sets (differentiable) 
    """
    n, num_classes = proba_values.shape

    tau = tau.to(device)

    sos = get_sos(num_classes)

    scores_all = smooth_aps_score_all(proba_values, sos=sos, device =device, dispersion=0.1, rng=None)
    
    #print(f'score_all_classes = {score_all_classes}')
    

    soft_set_preds = torch.sigmoid((tau.to(device) - scores_all)/T)
    # hard_set_preds = (proba_values <= tau.to(device)).to(torch.float32)
    
    hard_set_preds = (scores_all <= tau.to(device)).to(torch.float32)
    
    
    #print(f'T aps = {T}')
    #exit(1)

    if training_size:
        return torch.sum(set_preds, dim = 1) #torch.mean((score_all_classes - tau.to(device))/T).item()
    
    soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))
    hard_size_loss = torch.mean(torch.sum(hard_set_preds, dim = 1))

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - soft_set_preds) +  not_one_hot * soft_set_preds
    class_loss = torch.mean(torch.sum(class_set, dim = 1))

    #print(f'size_loss = {size_loss}, set_preds = {set_preds}')
    #exit()
    return soft_size_loss, hard_size_loss, class_loss 


def Estimate_size_loss_RAPS_hard(proba_values, Y_values, tau, device, num_classes = 100, T = 1.0, K = 0.0, lambda_RAPS = 0.01, k_RAPS = 5):
    """
    Compute the conformity scores and estimate the size of
    the conformal prediction sets (differentiable)
    """
    device = proba_values.device
    n, K = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
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
    prob_final_t = soft_indexing_all_labels(prob_sort_t, ranks_array_t)
    # Compute the conformity scores
    uu = torch.rand((n, K) ,dtype=float,device=device)


    """score_all_classes = prob_cum_t + prob_final_t * uu #Differentiable APS score for all classes"""

    score_all_classes = prob_cum_t + prob_final_t * uu + torch.maximum(lambda_RAPS * (ranks_array_t - k_RAPS), torch.zeros(ranks_array_t.shape).to(device)) #Differentiable RAPS score for all classes


    soft_set_preds = torch.sigmoid((tau.to(device) - score_all_classes)/T)
    hard_set_preds = (score_all_classes <= tau.to(device)).to(torch.float32)
    
    soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))
    hard_size_loss = torch.mean(torch.sum(hard_set_preds, dim = 1))

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - proba_values) +  not_one_hot * proba_values
    class_loss = torch.mean(torch.sum(class_set, dim = 1))

    return soft_size_loss, hard_size_loss, class_loss


def Estimate_size_loss_hard(output, scores, tau, Y_values, device, num_classes = 100, T = 1.0, K = 0.0, training_size = False):

    n = scores.shape[0]

    scores_tensor = torch.tensor(scores, device=device, dtype=output.dtype)

    soft_set_preds = torch.sigmoid((tau.to(device) - scores_tensor)/T)
    
    if training_size:
        return torch.sum(set_preds, dim = 1) 
    
    soft_size_loss = torch.mean(torch.maximum(torch.sum(soft_set_preds, dim = 1) - K, torch.zeros(n).to(device)))

    hard_set1 = (scores_tensor <= tau).to(torch.float32)
    hard_size_loss = hard_set1.sum(axis = 1).mean()

    one_hot = torch.nn.functional.one_hot(Y_values, num_classes = num_classes)
    not_one_hot = 1 - one_hot
    class_set = one_hot * (1 - soft_set_preds) +  not_one_hot * soft_set_preds
    class_loss = torch.mean(torch.sum(class_set, dim = 1))

    return soft_size_loss, hard_size_loss, class_loss


def maximize_for_h(optimizer_h, X, S, h, lambda_tensor, lambda_marginal, alpha, sigma = 0.1):
    h.train()
    optimizer_h.zero_grad()
    h_x = h(X)
    indicator_approx = 0.5 * (1 + torch.erf((-S + h_x) / (sigma * np.sqrt(2))))
    sum_lambda = torch.sum(lambda_tensor * X[:, :lambda_tensor.shape[0]] + lambda_marginal, axis=1)
    product = sum_lambda * (indicator_approx - (1.0 - alpha))
    h_x_positive = torch.clamp(h_x, min=0)
    loss_h = - torch.mean(product - h_x_positive)
    loss_h.backward()
    optimizer_h.step()

    return loss_h.item()


def minimize_for_h(optimizer_h, X, S, h, lambda_marginal, alpha, sigma=0.1):
    """
    Update 'h' to minimize the objective function g_alpha_n.
    """
    h.train()
    optimizer_h.zero_grad()

    X = X.view(X.size(0), -1) 

    # Compute model outputs
    h_x = h(X)

    # Smooth indicator approximation (tilde I)
    indicator_approx = 0.5 * (1 + torch.erf((-S + h_x) / (sigma * np.sqrt(2))))

    # Compute the first term: integration over Y (approximated by sum)
    integral_term = torch.mean(indicator_approx)

    # Compute the penalty term scaled by lambda_marginal
    penalty_term = lambda_marginal * ((1 - alpha) - torch.mean(indicator_approx))

    # Loss is the objective function
    loss_h = integral_term + penalty_term

    # Backpropagation
    loss_h.backward()
    optimizer_h.step()

    return loss_h.item()


def minimize_for_f(optimizer_lambda, X, S, h, lambda_marginal, alpha, sigma=0.1):
    """
    Update 'lambda_marginal' (dual variable) while keeping lambda_tensor constant.
    """
    optimizer_lambda.zero_grad()

    # Flatten X and detach to avoid graph reuse
    X = X.view(X.size(0), -1).detach()

    # Recompute h_x independently
    h_x = h(X)

    # Smooth indicator approximation (tilde I)
    indicator_approx = 0.5 * (1 + torch.erf((-S.detach() + h_x) / (sigma * np.sqrt(2))))

    # Compute the constraint violation term
    constraint_violation = (1 - alpha) - torch.mean(indicator_approx)

    # Detach lambda_marginal to ensure independence
    detached_lambda = lambda_marginal.detach().clone().requires_grad_(True)

    # Ensure lambda_marginal stays positive (projected gradient step)
    loss_lambda = -detached_lambda * constraint_violation

    # Backpropagation
    loss_lambda.backward()  # Ensure this is the only backward call for this graph
    optimizer_lambda.step()

    # Update lambda_marginal safely without affecting the graph
    lambda_marginal.data = torch.clamp(lambda_marginal.data + detached_lambda.grad.data, min=0)

    return lambda_marginal.item(), loss_lambda.item()

def CPL_loss(X, S, S_true, model, lambda_marginal, alpha, device, sigma=0.1):
    """
    Compute both h_loss and lambda_loss in a single function.
    
    Returns:
        loss_h: The loss for h optimization.
        loss_lambda: The loss for lambda optimization.
    """
    # if len(X.shape) > 2:  # If X has more than 2 dimensions (e.g., images)
    #     X = X.view(X.size(0), -1)  # Flatten input

    # X = X.view(X.size(0), -1)  # Flatten input
    
    # print(f"Input to forward_single: shape={X.shape}, requires_grad={X.requires_grad}")
    # model.module.eval()
    h_x = model.module.forward_single(X)
    # model.module.train()
    
    h_x_expand = h_x.unsqueeze(1).expand_as(S).clone()
    

    # Smooth indicator approximation (tilde I)
    indicator_approx = 0.5 * (1 + torch.erf((-S + h_x_expand) / (sigma * torch.sqrt(torch.tensor(2.0, device=device)))))
    
    # Sum over Y for each sample (dimension 1 corresponds to classes)
    sum_over_Y = torch.sum(indicator_approx, dim=1)  # Shape: (batch_size,)
    
    # Compute terms for h_loss
    indicator_approx_true = 0.5 * (1 + torch.erf((-S_true + h_x) / (sigma * np.sqrt(2))))
    
    mean_sum_over_Y = torch.mean(sum_over_Y)  # First term: 1/n \sum^n_{i=1} \sum_Y
    penalty_term = lambda_marginal * ((1 - alpha) - torch.mean(indicator_approx_true))  # Second term: Penalty
    loss_h = mean_sum_over_Y + penalty_term

    # Compute terms for lambda_loss

    hard_set1 = (S <= h_x_expand).to(torch.float32)
    hard_size_loss = hard_set1.sum(axis = 1).mean()

    constraint_violation = (1 - alpha) - torch.mean(indicator_approx_true)
    loss_lambda = -lambda_marginal * constraint_violation

    return loss_h, loss_lambda, mean_sum_over_Y, hard_size_loss, h_x   


