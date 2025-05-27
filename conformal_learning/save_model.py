import torch
import torchvision.models as models

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


from train import parser_file


sys.path.insert(0, './')

from conformal_learning.losses import LDAMLoss, FocalLoss


from conformal_learning.utils import evaluate_predictions, get_scores_HPS, get_scores, classwise_conformal, Marginal_conformal
from conformal_learning import black_boxes_CNN
from conformal_learning.utils import *
from conformal_learning.help import *
from conformal_learning.black_boxes_CNN import Estimate_quantile_n, Scores_RAPS_all_diff, Scores_APS_all_diff, Scores_HPS_all_diff, PinballMarginal, UniformMatchingLoss, Estimate_size_loss_RAPS, save_plot, find_scores_RAPS, find_scores_APS, find_scores_HPS, load_train_objs, base_path_for_finetune, load_checkpoint, prepare_dataloader, loss_fnc, check_path, create_final_data, create_folder, test_model, loss_cal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser_from_file = parser_file.initialize(parser)

    ags = parser_from_file.parse_args()

    if ags.arc == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes)  
    elif ags.arc == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  


    path = base_path_for_finetune(ags) #load_train_objs, base_path_for_finetune, load_checkpoint, prepare_dataloader, loss_fnc, check_path, create_final_data, create_folder, test_model
    print(f'path = {path}')

    if not os.path.exists(path):
        os.makedirs(path)
    save_path_1 = path+'/best_acc.pt'
    save_path_2 = path+'/best_loss.pt'
    save_path_3 = path+f'/final_epoch={ags.num_epochs}.pt'
    # Define the path where you want to save the model
    # save_path = '/your/special/path/resnet18.pth'

    # Save the model's state dictionary to the specified path
    torch.save(model.state_dict(), save_path_1)
    torch.save(model.state_dict(), save_path_2)
    torch.save(model.state_dict(), save_path_3)

    # print(f'Model saved to {save_path}')




