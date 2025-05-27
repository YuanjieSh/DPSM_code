import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns
from random import randint
import random


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def plot_scores_all(reg_loss, path1, label, TITLE = ''):
    if TITLE == 'Quantile':
        TITLE = 'COTI(VCCP)'

    if TITLE == 'Balanced':
        TITLE = 'Balanced(VCCP)'
    #print(f"title = {TITLE}")
    #level_adjusted = (1.0 - 0.1) * (1.0 + 1.0 / float(len(reg_loss)))
    #ind = int(int(0.9 * len(reg_loss)) + 1)
    #threshold = mquantiles(reg_loss, prob=level_adjusted)  
    plt.figure(figsize=(10,6))

    sns.set_style("darkgrid")
    if not os.path.exists(path1):
        os.makedirs(path1)
    #for i in range(len(reg_loss)):
    min_length = np.min([len(li) for li in reg_loss])

    new = []
    for i in range(len(reg_loss)):
        new.append(reg_loss[i][:min_length])
    reg_loss1 = np.array(new)


    sd = np.std(reg_loss1, axis = 0)
    reg_l = np.mean(reg_loss1, axis = 0)
    indcs = np.argsort(reg_l)
    sd = sd[indcs]
    reg_l = reg_l[indcs]

    level_adjusted = (1.0 - 0.1) * (1.0 + 1.0 / float(len(reg_l)))
    ind = int(int(0.9 * len(reg_l)) + 1)
    threshold = mquantiles(reg_l, prob=level_adjusted)  
    xticks_font = 15
    yticks_font = 30
    xy_font = 25
    plt.plot(np.arange(len(reg_l)), reg_l, color = 'red')
    # Visualize the result


    plt.fill_between(np.arange(len(reg_l)), reg_l - sd, reg_l + sd,
                 color='gray', alpha=0.8)

    plt.xticks(fontsize = xticks_font)
    plt.yticks(fontsize = yticks_font)
    plt.ylabel('APS scores', fontsize = xy_font)
    plt.xlabel('Number of points', fontsize = xy_font)
    #plt.title('Class = {}'.format(label), fontsize = 40)

    #plt.axhline(0.9, ls='--', c='black', linewidth = 4)
    plt.axvline(ind, ls='--', c='lime', linewidth = 4)

    plt.axhline(threshold, ls='--', c='green', linewidth = 4)
    plt.ylim(-0.01, 1.01)
    plt.savefig(path1 + 'label1={}.png'.format(label), dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')   

    random.seed(0)

    colors = []
    for i in range(10):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    res = pd.DataFrame()
    vals, thresholds = [], []
    for i in range(len(reg_loss)):
        reg_l = reg_loss[i][:min_length]
        level_adjusted = (0.1) * (1.0 + 1.0 / float(len(reg_l)))
        ind = int(int(0.9 * len(reg_l)) + 1)
        ind1 = int(int(0.85 * len(reg_l)) + 1)
        ind2 = int(int(0.95 * len(reg_l)))
        #reg_l = 1 - np.sort(-reg_l)
        reg_l = 1 - np.sort(reg_l)
        threshold = mquantiles(reg_l, prob=level_adjusted)  
        thresholds.append(threshold[0])
        #print(f"threshold = {threshold}, {threshold[0]}")
        xticks_font = 17.5
        yticks_font = 25
        xy_font = 25
        plt.plot(np.arange(len(reg_l)), reg_l, '*', color = colors[i])
        plt.xticks(fontsize = xticks_font)
        plt.yticks(fontsize = yticks_font)
        plt.ylabel('APS scores', fontsize = xy_font)
        plt.xlabel('Number of points', fontsize = xy_font)
        plt.title(TITLE, fontsize = 40)

        #plt.axhline(0.9, ls='--', c='black', linewidth = 4)
        plt.axvline(ind, ls='--', c='lime', linewidth = 4)

        #plt.axhline(threshold, ls='--', c=colors[i], linewidth = 4)
        res1 = pd.DataFrame({'Label:'.format(i):i, 'Thr': threshold})
        res = pd.concat([res, res1])
        #plt.ylim(0.75, 1.01)
        #plt.xlim(ind1, ind2)
        vals.append(reg_l[ind])

    #print(f"thresholds = {thresholds}")

    mean = np.mean(vals)
    std = np.std(vals)
    min_threshold = np.min(thresholds)
    max_threshold = np.max(thresholds)
    #print(f" thresholds = {min_threshold}, {max_threshold}")
    #plt.title(label = 'Me.={} Mean={} Std={}'.format(TITLE, round(mean,3), round(std, 4)), fontsize = 20)
    plt.title(label = TITLE, fontsize = 20)
    plt.axhline(min_threshold, ls='--', c='black', linewidth = 4)
    plt.axhline(max_threshold, ls='--', c='black', linewidth = 4)

    #print(f"res = {res}")
    #exit(1)


    res.to_csv(path1 + 'thresholds_class.csv', index = False)
    #plt.ylim(0.1, 0.2)
    plt.savefig(path1 + 'method={}.png'.format(TITLE), dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')

    
def plot_scores(reg_loss, path1, label):
    level_adjusted = (1.0 - 0.1) * (1.0 + 1.0 / float(len(reg_loss)))
    ind = int(int(0.9 * len(reg_loss)) + 1)
    threshold = mquantiles(reg_loss, prob=level_adjusted)  
    plt.figure(figsize=(10,6))

    sns.set_style("darkgrid")

    if not os.path.exists(path1):
        os.makedirs(path1)
    xticks_font = 15
    yticks_font = 30
    xy_font = 25
    plt.plot(np.arange(len(reg_loss)), np.sort(reg_loss), '*', color = 'red')
    plt.xticks(fontsize = xticks_font)
    plt.yticks(fontsize = yticks_font)
    plt.ylabel('APS scores', fontsize = xy_font)
    plt.xlabel('Number of points', fontsize = xy_font)
    #plt.title('Class = {}'.format(label), fontsize = 40)

    plt.axhline(0.9, ls='--', c='black', linewidth = 4)
    plt.axvline(ind, ls='--', c='lime', linewidth = 4)

    plt.axhline(threshold, ls='--', c='green', linewidth = 4)

    plt.ylim(-0.01, 1.01)
    #plt.xlim(-0.01, 1.01)

    plt.savefig(path1 + 'label={}.png'.format(label), dpi = 100, bbox_inches='tight', pad_inches=.1)
    plt.close('all')


def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = torch.nn.CrossEntropyLoss().cuda()

    T = torch.nn.Parameter(torch.Tensor([1.3]).cuda())

    optimizer = torch.optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        #print(f"T = {T} iter = {iter}")
        if abs(T_old - T.item()) < epsilon:
            break
    return T 


def get_logits_targets(model, loader, num_classes, device):
    #print(model)
    #exit(1)
    #print(f"model device = {next(model.parameters()).get_device()}")
    device = next(model.parameters()).get_device()

    logits = torch.zeros((len(loader.dataset), num_classes)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    #print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in loader:
            
            batch_logits = model(x.to(device)).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    #dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return torch.argmax(logits, dim = 1).long(), logits, labels.long()


def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)

def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def generalized_inverse_quantile_score(probabilities, labels, u=None, all_combinations=True):

    # whether to do a randomized score or not
    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # sort probabilities from high to low
    sorted_probabilities = -np.sort(-probabilities)

    # create matrix of cumulative sum of each row
    cumulative_sum = np.cumsum(sorted_probabilities, axis=1)

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[np.arange(num_of_points), label_ranks.T].T

    # compute the probability of the last label that enters
    last_label_prob = sorted_probabilities[np.arange(num_of_points), label_ranks.T].T

    # remove the last label probability or a multiplier of it in the randomized score
    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - np.diag(u) @ last_label_prob
    return scores

def get_scores(model, DataLoader, num_classes = 10, device='cpu', GPU_CAPACITY=1024, all_combinations = True):
    #fx, y = get_logits_targets(model, DataLoader, BatchSize = GPU_CAPACITY, NumClass = num_classes, device = device)
    layer_prob = torch.nn.Softmax(dim=1)

    y_pred, logits, y_true= get_logits_targets(model, DataLoader, num_classes, device)

    #ratio = 0.2
    #Temp_x = logits[:int(ratio * len(y_pred))]
    #Temp_y = y_true[:int(ratio * len(y_pred))]
    #Temp_data = torch.utils.data.TensorDataset(Temp_x, Temp_y)
    #Temp_loader = torch.utils.data.DataLoader(Temp_data, batch_size = 32)

    #T = platt_logits(Temp_loader, max_iters=10, lr=0.01, epsilon=0.01)
    #T = T.detach().to('cpu')
    #print(f"T1 = {T}")
    #exit(1)
    T = torch.tensor([1.0])
    #print(f"T2 = {T}")
    #exit(1)
    #print(f"device = {logits.get_device()}, {T.get_device()}")
    #exit(1)
    simple_outputs = layer_prob(logits/T).detach().numpy()

    n = len(y_true)
    #print(f"n  = {n}")
    #exit(1)
    # create container for the scores
    #scores_simple = np.zeros((0, n, num_classes))
    rng = default_rng(seed = 0)
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)
    #print(f"shape = {simple_outputs.shape} {y_true.shape}, {n} {uniform_variables.shape}")

    scores_simple = generalized_inverse_quantile_score(simple_outputs, np.arange(num_classes), uniform_variables, all_combinations=all_combinations)
    #print(f"shape = {scores_simple.shape}")
    return scores_simple, y_true, y_pred


def generalized_inverse_quantile_score_RAPS(probabilities, labels, u=None, lambda_RAPS =.01, k_RAPS =5, all_combinations=True):

    # whether to do a randomized score or not
    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # sort probabilities from high to low
    sorted_probabilities = -np.sort(-probabilities)

    # create matrix of cumulative sum of each row
    cumulative_sum = np.cumsum(sorted_probabilities, axis=1)

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[np.arange(num_of_points), label_ranks.T].T

    ranks = label_ranks + 1

    # compute the probability of the last label that enters
    last_label_prob = sorted_probabilities[np.arange(num_of_points), label_ranks.T].T

    penalty = lambda_RAPS * np.maximum(0, ranks - k_RAPS)

    scores = scores + penalty

    # remove the last label probability or a multiplier of it in the randomized score
    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - np.diag(u) @ last_label_prob
    return scores

def get_scores_RAPS(model, DataLoader, num_classes = 10, lambda_RAPS =.01, k_RAPS =5, device='cpu', GPU_CAPACITY=1024, all_combinations = True):
    #fx, y = get_logits_targets(model, DataLoader, BatchSize = GPU_CAPACITY, NumClass = num_classes, device = device)
    layer_prob = torch.nn.Softmax(dim=1)

    y_pred, logits, y_true= get_logits_targets(model, DataLoader, num_classes, device)

    #ratio = 0.2
    #Temp_x = logits[:int(ratio * len(y_pred))]
    #Temp_y = y_true[:int(ratio * len(y_pred))]
    #Temp_data = torch.utils.data.TensorDataset(Temp_x, Temp_y)
    #Temp_loader = torch.utils.data.DataLoader(Temp_data, batch_size = 32)

    #T = platt_logits(Temp_loader, max_iters=10, lr=0.01, epsilon=0.01)
    #T = T.detach().to('cpu')
    #print(f"T1 = {T}")
    #exit(1)
    T = torch.tensor([1.0])
    #print(f"T2 = {T}")
    #exit(1)
    #print(f"device = {logits.get_device()}, {T.get_device()}")
    #exit(1)
    simple_outputs = layer_prob(logits/T).detach().numpy()

    n = len(y_true)
    #print(f"n  = {n}")
    #exit(1)
    # create container for the scores
    #scores_simple = np.zeros((0, n, num_classes))
    rng = default_rng(seed = 0)
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)
    #print(f"shape = {simple_outputs.shape} {y_true.shape}, {n} {uniform_variables.shape}")

    scores_simple = generalized_inverse_quantile_score_RAPS(simple_outputs, np.arange(num_classes), uniform_variables, lambda_RAPS =.01, k_RAPS =5, all_combinations=all_combinations)
    #print(f"shape = {scores_simple.shape}")
    return scores_simple, y_true, y_pred

def calibration(scores_simple=None, alpha = 0.1, path1 = './'):
    # size of the calibration set
    #print(f"scores_simple = {scores_simple.shape}")
    n_calib = scores_simple.shape[0]


    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))

    thresholds = mquantiles(scores_simple, prob=level_adjusted)            
    plot_scores(scores_simple, path1, 'ALL')

    return thresholds, pd.DataFrame({'Quantile': thresholds, 'Label': ['All']})

def PredictionICP(scores_simple=None, thresholds=None, y_pred = None):
    n = scores_simple.shape[0]
    predicted_sets = []

    S_hat_simple = [np.where(scores_simple[i, :] <= thresholds[y_pred[i]])[0].tolist() for i in range(n)]

    predicted_sets.append(S_hat_simple)      

    return predicted_sets


def prediction(scores_simple=None, thresholds=None, Method = None, calibration = 'VCP', y_hat = None):


    n = scores_simple.shape[0]
    predicted_sets = []

    if calibration != 'ICP':
        

        S_hat_simple = [np.where(scores_simple[i, :] <= thresholds)[0].tolist() for i in range(n)]

        #if Method == 'Quantile':
        #for k in range(len(S_hat_simple)):
        #    if len(S_hat_simple[k]) == 0:
        #        S_hat_simple[k] = [np.argsort(scores_simple[k, :])[0]]
        #        #print(f"K = {k}") 
        #    predicted_sets.append(S_hat_simple)   
                

        #else:
        predicted_sets.append(S_hat_simple)  
        return predicted_sets

    elif calibration == 'ICP':
        return PredictionICP(scores_simple=scores_simple, thresholds=thresholds, y_pred = y_hat)


    else:
        raise ValueError("Please specify the calibration type")

    #print(f"Method = {Method}, predicted_sets = {predicted_sets}")      




def predictions_last(S, y, y_pred):

    label = np.unique(y)

    if len(label) == 1:
      label = label[0]
    elif len(label) > 1:
      label = "All"


    # count
    count = len(y)
    #print(y.shape, 'y shape', label)
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # size of prediction set
    size = [len(S[i]) for i in range(len(y))]
    # average size
    size_mean = np.mean([len(S[i]) for i in range(len(y))])
    # median size
    size_median = np.median([len(S[i]) for i in range(len(y))])
    # size conditional on coverage
    # which prediction sets cover the true label
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    # average size of sets that cover true y
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # coverage indicator
    cover_indicator = [y[i] in S[i] for i in range(len(y))]
    # correlation between size of prediction set and coverage indicator
    corr_size_cover = np.corrcoef(cover_indicator, size)
    single_corr = corr_size_cover[0][1]
    # test accuracy
    #print(f"y = {y} {y_pred}")
    test_acc = (torch.sum(y == y_pred)/len(y)).item()
    # Combine results
    out = pd.DataFrame({'Label': [label],
                        'Count': [count],
                        'Coverage': [coverage],
                        'Size (mean)': [size_mean], 
                        'Size (median)': [size_median], 
                        'Size cover': [size_cover],
                        # 'Corr size-cover': [corr_size_cover],
                        'Single Corr':[single_corr],
                        'Test Acc':[test_acc]})


    return out

def evaluate_predictions(S, y, y_pred, coverage_on_label=False, num_of_classes=10):
    #y = torch.from_numpy(y)
    #y_pred = torch.from_numpy(y_pred)

    #print(f"S = {S}")
    if num_of_classes == 27:
        start, end = 1, num_of_classes
    else:
        start, end = 0, num_of_classes
    if coverage_on_label:
        out = pd.DataFrame()
        for i in range(start, end):
            i_idx = torch.where(y == i)[0].numpy().tolist()
            #print(f"i_idx = {i_idx}")
            out_i = predictions_last([S[k] for k in i_idx], y[i_idx], y_pred[i_idx])
            out = pd.concat([out, out_i])
    else:
        out = predictions_last(S, y, y_pred)

    return out



def calibration_classwise(scores_simple=None, alpha=0.1, y = None, path1 = './', all = False, method = 'Balanced'):
    thresholds = 0
    #y = torch.from_numpy(y)
    scores_all = []
    THR = pd.DataFrame()
    if y is not None:
        classes = np.unique(y)
        for cls in classes:
            Indices_cls = torch.where(y == cls)[0]
            level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(len(Indices_cls)))

            q = mquantiles(np.array(scores_simple[Indices_cls]), prob=level_adjusted)
            plot_scores(np.array(scores_simple[Indices_cls]), path1, cls)
            scores_all.append(np.array(scores_simple[Indices_cls]))
            if q > thresholds:
                thresholds = q
            all_thresholds = pd.DataFrame({'Quantile': q, 'Label': [cls]})
            THR = pd.concat([THR, all_thresholds])
    
    if all:
        plot_scores_all(scores_all, path1, 'all', method)

    return thresholds, THR




def CalibrationVCP(scores_simple=None, alpha = 0.1, y = None, y_hat = None, path1 = './'):

    n_all = len(scores_simple)

    cal_idx, hyper_idx = train_test_split(np.arange(n_all), test_size=0.01, random_state = 0)
    y_cal, y_hyper = y[cal_idx], y[hyper_idx]
    y_hat_cal, y_hat_hyper = y_hat[cal_idx], y_hat[hyper_idx]

    scores_simple_cal, scores_simple_hyper = scores_simple[cal_idx, y_cal], scores_simple[hyper_idx, :]

    n_calib = scores_simple_cal.shape[0]

    alpha_l, alpha_u = 0.0, 1.0

    i = 0
    while alpha_l < alpha_u:
        alpha_eff = (alpha_l + alpha_u)/2

        # Compute thresholds
        level_adjusted = (alpha_eff) * (1.0 + 1.0 / float(n_calib))

        q = mquantiles(scores_simple_cal, prob=level_adjusted)     
        predicted_clean_sets_base = prediction(scores_simple=scores_simple_hyper, thresholds=q)
        #print(f"predicted_clean_sets_base = {np.array(predicted_clean_sets_base).shape}")
        res = evaluate_predictions(predicted_clean_sets_base[0], y_hyper, y_hat_hyper, coverage_on_label=False, num_of_classes=10)
        cover = res['Coverage'].to_numpy()[0]
        #print(f"coverage = {cover}, {alpha_l}, {alpha_u}, {alpha_eff}")

        if cover >= 1 - alpha and (cover - 1 + alpha < 0.015):
            thresholds = q
            break
        elif cover > 1 - alpha and (cover - 1 + alpha > 0.015):
            alpha_u = alpha_eff
        elif cover < 1 - alpha:
            alpha_l = alpha_eff
        elif i > 15:
            break
        else:
            raise ValueError("Need more conditions")
        i += 1
    #plot_scores(scores_simple, path1, 'ALL')

    return thresholds, pd.DataFrame({'Quantile': thresholds, 'Label': ['All']})


def CalibrationICP1(scores_simple=None, alpha=0.1, y = None, y_hat = None, path1 = './', all = True, method = 'Balanced'):

    n_all = len(scores_simple)

    cal_idx, hyper_idx = train_test_split(np.arange(n_all), test_size=0.01, random_state = 0)
    y_cal, y_hyper = y[cal_idx], y[hyper_idx]
    y_hat_cal, y_hat_hyper = y_hat[cal_idx], y_hat[hyper_idx]

    scores_simple_cal, scores_simple_hyper = scores_simple[cal_idx, y_cal], scores_simple[hyper_idx, :]

    thresholds = []
    #y = torch.from_numpy(y)
    scores_all = []
    THR = pd.DataFrame()
    if y is not None:
        classes = np.unique(y_cal)
        for cls in classes:
            Indices_cls_cal = torch.where(y_cal == cls)[0]
            Indices_cls_hyper = torch.where(y_hyper == cls)[0]

            alpha_l, alpha_u = 0.0, 1.0

            i = 0
            while alpha_l < alpha_u:
                alpha_eff = (alpha_l + alpha_u)/2
                level_adjusted = (1.0 - alpha_eff) * (1.0 + 1.0 / float(len(Indices_cls_cal)))

                q = mquantiles(np.array(scores_simple_cal[Indices_cls_cal]), prob=level_adjusted)
                predicted_clean_sets_base = PredictionICP(scores_simple=scores_simple_hyper[Indices_cls_hyper], thresholds=q)
                #print(f"predicted_clean_sets_base = {np.array(predicted_clean_sets_base).shape}")
                res = evaluate_predictions(predicted_clean_sets_base[0], y_hyper[Indices_cls_hyper], y_hat_hyper[Indices_cls_hyper], coverage_on_label=False, num_of_classes=10)
                cover = res['Coverage']

                if cover > 1 - alpha and (cover - 1 + alpha < 0.015):
                    thresholds.append(q)
                    break
                elif cover > 1 - alpha and (cover - 1 + alpha > 0.015):
                    alpha_u = alpha_eff
                elif cover < 1 - alpha:
                    alpha_l = alpha_eff
                elif i > 15:
                    break
                else:
                    raise ValueError("Need more conditions")
                i += 1

            scores_all.append(np.array(scores_simple_cal[Indices_cls_cal]))

            all_thresholds = pd.DataFrame({'Quantile': q, 'Label': [cls]})
            THR = pd.concat([THR, all_thresholds])
    
    if all:
        plot_scores_all(scores_all, path1, 'all', method)

    return thresholds, THR

def ClassAllQuantile(classes, alpha_eff, y, scores_simple_cal):
    thresholds = []
    for cls in classes:
        Indices_cls = torch.where(y == cls)[0]
        level_adjusted = (alpha_eff[cls]) * (1.0 + 1.0 / float(len(Indices_cls)))

        q = mquantiles(np.array(scores_simple_cal[Indices_cls]), prob=level_adjusted)
        thresholds.append(q)
    return np.array(thresholds)

def CalibrationICP(scores_simple=None, alpha=0.1, y = None, y_hat = None, path1 = './', all = True, method = 'Balanced', hypertune = False):


    n_all = len(scores_simple)

    cal_idx, hyper_idx = train_test_split(np.arange(n_all), test_size=0.01, random_state = 0)
    y_cal, y_hyper = y[cal_idx], y[hyper_idx]
    y_hat_cal, y_hat_hyper = y_hat[cal_idx], y_hat[hyper_idx]

    scores_simple_cal, scores_simple_hyper = scores_simple[cal_idx, y_cal], scores_simple[hyper_idx, :]

    if hypertune:
        thresholds = []
        #y = torch.from_numpy(y)
        scores_all = []
        THR = pd.DataFrame()
        if y is not None:
            classes = np.unique(y_cal)

            i = 0
            alpha_l = np.array([0.0 for _ in range(10)])
            alpha_u = np.array([1.0 for _ in range(10)])
            while i < 10:

                alpha_eff = (alpha_l + alpha_u)/2



                thresholds = ClassAllQuantile(classes, alpha_eff, y_cal, scores_simple_cal)
            
                predicted_clean_sets_base = PredictionICP(scores_simple=scores_simple_hyper, thresholds=thresholds, y_pred = y_hat_hyper )
                #print(f"predicted_clean_sets_base = {np.array(predicted_clean_sets_base).shape}")
                res = evaluate_predictions(predicted_clean_sets_base[0], y_hyper, y_hat_hyper, coverage_on_label=True, num_of_classes=10)
                coverages = torch.from_numpy(res['Coverage'].to_numpy())
                indices_l = torch.where(coverages < 1 - alpha)[0]
                alpha_l[indices_l] = alpha_eff[indices_l]
                indices_u = torch.where(coverages >= 1 - alpha)[0]
                alpha_u[indices_u] = alpha_eff[indices_u]
                i +=1 

    else:
        THR = pd.DataFrame()

        alpha_eff = np.array([1 - alpha for _ in range(10)])
        classes = np.unique(y_cal)
        if method == 'Balanced':
            thresholds = ClassAllQuantile(classes, alpha_eff + 0.03, y_cal, scores_simple_cal)
        elif method == 'Cross-entropy':
            #print('CE')
            thresholds = ClassAllQuantile(classes, alpha_eff + 0.03, y_cal, scores_simple_cal)



    return thresholds, THR


def CalibrationVCCP(scores_simple=None, alpha=0.1, y = None, y_hat = None, path1 = './', all = False, method = 'Balanced', hypertune = False):

    n_all = len(scores_simple)

    cal_idx, hyper_idx = train_test_split(np.arange(n_all), test_size=0.01, random_state = 0)
    y_cal, y_hyper = y[cal_idx], y[hyper_idx]
    y_hat_cal, y_hat_hyper = y_hat[cal_idx], y_hat[hyper_idx]

    scores_simple_cal, scores_simple_hyper = scores_simple[cal_idx, y_cal], scores_simple[hyper_idx, :]
    #y = torch.from_numpy(y)
    scores_all = []
    THR = pd.DataFrame()

    if hypertune:
        if y is not None:
            classes = np.unique(y_cal)

            alpha_l, alpha_u = 0.0, 1.0

            i = 0
            while alpha_l < alpha_u:
                thresholds = 0

                alpha_eff = (alpha_l + alpha_u)/2
                for cls in classes:
                    Indices_cls = torch.where(y_cal == cls)[0]
                    level_adjusted = (alpha_eff) * (1.0 + 1.0 / float(len(Indices_cls)))

                    q = mquantiles(np.array(scores_simple_cal[Indices_cls]), prob=level_adjusted)
                    #plot_scores(np.array(scores_simple[Indices_cls]), path1, cls)
                    scores_all.append(np.array(scores_simple_cal[Indices_cls]))
                    if q > thresholds:
                        thresholds = q
                    all_thresholds = pd.DataFrame({'Quantile': q, 'Label': [cls]})
                    THR = pd.concat([THR, all_thresholds])
            
                predicted_clean_sets_base = prediction(scores_simple=scores_simple_hyper, thresholds=thresholds)
                #print(f"predicted_clean_sets_base = {np.array(predicted_clean_sets_base).shape}")
                res = evaluate_predictions(predicted_clean_sets_base[0], y_hyper, y_hat_hyper, coverage_on_label=True, num_of_classes=10)
                cover = np.min(res['Coverage'].to_numpy())
                print(f"cover = {cover}, alpha_l = {alpha_l}, alpha_u = {alpha_u}, alpha_eff = {alpha_eff}")

                if cover >= 1 - alpha and (cover - 1 + alpha < 0.02):
                    thresholds = thresholds
                    break
                elif cover > 1 - alpha and (cover - 1 + alpha > 0.02):
                    alpha_u = alpha_eff
                elif cover < 1 - alpha:
                    alpha_l = alpha_eff
                elif i > 15:
                    break
                else:
                    raise ValueError("Need more conditions")
                i += 1

        if all:
            plot_scores_all(scores_all, path1, 'all', method)
    else:
        scores_all = []
        alpha_eff = 1 - alpha
        classes = np.unique(y_cal)
        thresholds = 0
        for cls in classes:
            Indices_cls = torch.where(y_cal == cls)[0]
            level_adjusted = (alpha_eff+0.03) * (1.0 + 1.0 / float(len(Indices_cls)))

            q = mquantiles(np.array(scores_simple_cal[Indices_cls]), prob=level_adjusted)
            #plot_scores(np.array(scores_simple[Indices_cls]), path1, cls)
            scores_all.append(np.array(scores_simple_cal[Indices_cls]))
            if q > thresholds:
                thresholds = q
            all_thresholds = pd.DataFrame({'Quantile': q, 'Label': [cls]})
            THR = pd.concat([THR, all_thresholds])
    return thresholds, THR


#################################
#      Classwise prediction
###############################

def get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf):
    '''
    Compute the quantities necessary for performing conformal prediction with exact coverage
    
    Inputs:
        score: length-n array of conformal scores
        alpha: float between 0 and 1 denoting the desired coverage level
        default_qhat = value used when n is too small for computing the relevant quantiles
    Outputs:
        q_a: The smallest score that is larger than (n+1) * (1-alpha) of the other scores (= normal conformal qhat)
        q_b: The smallest score that is larger than (n+1) * (1-alpha) - 1 of the other scores
        gamma: value between 0 and 1 such that gamma * q_a + (1-gamma)*g_b = 1-alpha
    '''

    n = len(scores)
    
    if n == 0:
        return np.inf, np.inf, 1
    
    val_a = np.ceil((n+1)*(1-alpha)) / n
    if val_a > 1:
        q_a = default_qhat
    else:
        q_a = np.quantile(scores, val_a)#, method='inverted_cdf')
        
    val_b = (np.ceil((n+1)*(1-alpha)-1)) / n
    if val_b > 1:
        q_b = default_qhat
        
    else:
        q_b = np.quantile(scores, val_b)#, method='inverted_cdf') 
        
    if val_a > 1 and val_b > 1:
        gamma = 1 # Any value would work, since q_a and q_b are both equal to default_qhat
    else:
        overcov = np.ceil((n+1)*(1-alpha))/(n+1) - (1-alpha) # How much coverage using q_a will exceed 1 - alpha
        undercov = (1-alpha) - (np.ceil((n+1)*(1-alpha))-1)/(n+1)  #  How much coverage using q_b will undershoot 1 - alpha
        gamma = undercov / (undercov + overcov)

    return q_a, q_b, gamma


def compute_coverage(true_labels, set_preds):
    true_labels = np.array(true_labels) # Convert to numpy to avoid weird pytorch tensor issues
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc


def compute_class_specific_coverage(true_labels, set_preds):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        #print(f"idx = {idx}")
        selected_preds = [set_preds[i] for i in idx]
        #print(f"selected_preds = {selected_preds}")
        num_correct = np.sum([1 if np.any(np.array(pred_set) == k) else 0 for pred_set in selected_preds])
        #print(f"num_correct = {num_correct}")
        #exit(1)
        class_specific_cov[k] = num_correct / len(selected_preds)
        
    return class_specific_cov


def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov = compute_class_specific_coverage(val_labels, preds)

    #print(f"class_cond_cov = {class_cond_cov}")
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average gap for classes that are over-covered
    overcov_idx = (class_cond_cov > (1-alpha))
    overcov_gap = np.mean(class_cond_cov[overcov_idx] - (1-alpha))

    # Average gap for classes that are under-covered
    undercov_idx = (class_cond_cov < (1-alpha))
    undercov_gap = np.mean(np.abs(class_cond_cov[undercov_idx] - (1-alpha)))
    
    # Fraction of classes that are at least 10% under-covered
    thresh = .1
    very_undercovered = np.mean(class_cond_cov < (1-alpha-thresh))
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'undercov_gap': undercov_gap, 
                         'overcov_gap': overcov_gap, 
                         'max_gap': max_gap,
                         'very_undercovered': very_undercovered,
                         'marginal_cov': marginal_cov,
                         'raw_class_coverages': class_cond_cov,
                         'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'mean': np.mean(curr_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])}
    
    #print('CLASS COVERAGE GAP:', avg_class_cov_gap)
    #print('AVERAGE SET SIZE:', np.mean(curr_set_sizes))
    
    return class_cov_metrics, set_size_metrics

def compute_exact_coverage_class_specific_params(scores_all, labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_params=None):
    '''
    Compute the quantities necessary for performing classwise conformal prediction with exact coverage
    
    Inputs:
        - scores_all: (num_instances, num_classes) array where scores_all[i,j] = score of class j for instance i
        - labels: (num_instances,) array of true class labels
        - num_classes: Number of classes
        - alpha: number between 0 and 1 specifying desired coverage level
        - default_qhat: Quantile that will be used when there is insufficient data to compute a quantile
        - null_params: Dict of {'q_a': ..., 'q_b': ...., 'gamma', ...} to be assigned to
               class -1 (will be appended to end of param lists). Not needed if -1 does not appear in labels
    '''
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        
        idx = (labels == k)
        if len(scores_all.shape) == 2:
            scores = scores_all[idx, k]
        else:
            scores = scores_all[idx]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=default_qhat)
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    if -1 in labels:
        q_as = np.concatenate((q_as, [null_params['q_a']]))
        q_bs = np.concatenate((q_bs, [null_params['q_b']]))
        gamma = np.concatenate((gammas, [null_params['gamma']]))
        
    return q_as, q_bs, gammas

def get_conformal_quantile(scores, alpha, default_qhat=np.inf, exact_coverage=False):
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
            qhat = np.quantile(scores, val)#, method='inverted_cdf')

        return qhat

def compute_qhat(scores_all, true_labels, alpha, exact_coverage=False, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        - true_labels: num_instances length array of ground truth labels
        - alpha: float between 0 and 1 specifying coverage level
        - plot_scores: If True, plot histogram of true class scores 
    '''
    # If necessary, select scores that correspond to correct label
    if len(scores_all.shape) == 2:
        scores = np.squeeze(np.take_along_axis(scores_all, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = scores_all
    
    q_hat = get_conformal_quantile(scores, alpha, exact_coverage=exact_coverage)
    
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hat

def reconformalize(qhats, scores, labels, alpha, adjustment_min=-1, adjustment_max=1):
    '''
    Adjust qhats by additive factor so that marginal coverage of 1-alpha is achieved
    '''
    print('Applying additive adjustment to qhats')
    # ===== Perform binary search =====
    # Convergence criteria: Either (1) marginal coverage is within tol of desired or (2)
    # quantile_min and quantile_max differ by less than .001, so there is no need to try 
    # to get a more precise estimate
    tol = 0.0005

    marginal_coverage = 0
    while np.abs(marginal_coverage - (1-alpha)) > tol:

        adjustment_guess = (adjustment_min +  adjustment_max) / 2
        print(f"\nCurrent adjustment: {adjustment_guess:.6f}")

        curr_qhats = qhats + adjustment_guess 

        preds = create_classwise_prediction_sets(scores, curr_qhats)
        marginal_coverage = compute_coverage(labels, preds)
        print(f"Marginal coverage: {marginal_coverage:.4f}")

        if marginal_coverage > 1 - alpha:
            adjustment_max = adjustment_guess
        else:
            adjustment_min = adjustment_guess
        print(f"Search range: [{adjustment_min}, {adjustment_max}]")

        if adjustment_max - adjustment_min < .00001:
            adjustment_guess = adjustment_max # Conservative estimate, which ensures coverage
            print("Adequate precision reached; stopping early.")
            break
            
    print('Final adjustment:', adjustment_guess)
    qhats += adjustment_guess
    
    return qhats 

def construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        set_preds.append(np.where(scores_all[i,:] <= q_hats)[0]) 

    return set_preds


def compute_class_specific_qhats(cal_scores_all, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all: 
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed). 
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the 
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat. 
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1 
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for 
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhat = get_exact_coverage_conformal_params(cal_scores_all, alpha, default_qhat=default_qhat)
        q_a, q_b, gamma = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, 
                                                                       num_classes, alpha, 
                                                                       default_qhat=default_qhat, null_params=null_qhat)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha=alpha)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k)
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alpha, default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))


        # Optionally apply shrinkage 
        if regularize:
            N = num_classes
            n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
            shrinkage_factor = .03 * n_k # smaller = less shrinkage
            shrinkage_factor = np.minimum(shrinkage_factor, 1)
            print('SHRINKAGE FACTOR:', shrinkage_factor)  
            print(np.min(shrinkage_factor), np.max(shrinkage_factor))
            q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)

            # Correct qhats via additive factor to achieve marginal coverage
            q_hats = reconformalize(q_hats, reserved_scores, reserved_labels, alpha)


        return q_hats

def compute_marginal_qhats(cal_scores_all, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all: 
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed). 
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the 
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat. 
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1 
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for 
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhat = get_exact_coverage_conformal_params(cal_scores_all, alpha, default_qhat=default_qhat)
        q_a, q_b, gamma = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, 
                                                                       num_classes, alpha, 
                                                                       default_qhat=default_qhat, null_params=null_qhat)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha=alpha)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class

            q_hats[k] = get_conformal_quantile(cal_scores_all, alpha, default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))


        # Optionally apply shrinkage 
        if regularize:
            N = num_classes
            n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
            shrinkage_factor = .03 * n_k # smaller = less shrinkage
            shrinkage_factor = np.minimum(shrinkage_factor, 1)
            print('SHRINKAGE FACTOR:', shrinkage_factor)  
            print(np.min(shrinkage_factor), np.max(shrinkage_factor))
            q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)

            # Correct qhats via additive factor to achieve marginal coverage
            q_hats = reconformalize(q_hats, reserved_scores, reserved_labels, alpha)


        return q_hats
    

# Create classwise prediction sets
def create_classwise_prediction_sets(scores_all, q_hats, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            set_preds.append(np.where(scores_all[i,:] <= q_hats)[0].tolist())

    return set_preds

# Classwise conformal pipeline
def Marginal_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
    See compute_class_specific_qhats() docstring for more details about expected inputs.
    '''
    
    classwise_qhats = compute_marginal_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)

    #print(f"classwise_preds = {classwise_preds}")
    #print(f"classwise_qhats = {classwise_qhats}")
    #exit(1)
    curr_set_sizes = [len(x) for x in classwise_preds]
    avg_set_sizes = np.mean(curr_set_sizes)
    if avg_set_sizes <=1:
        classwise_preds_new = remove_empty_set(classwise_preds, val_scores_all)
        classwise_preds = classwise_preds_new
    #coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)

    THR = pd.DataFrame()
    for cls in np.arange(num_classes):
        all_thresholds = pd.DataFrame({'Quantile': classwise_qhats[cls], 'Label': [cls]})
        THR = pd.concat([THR, all_thresholds])

    #print(f"set_size_metrics = {set_size_metrics}, coverage_metrics = {coverage_metrics}, classwise_preds = {classwise_preds}")
    #exit(1)
    #print(f"classwise_qhats = {classwise_qhats}")

    return [np.max(classwise_qhats)], THR, classwise_preds


# Classwise conformal pipeline
def classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
    See compute_class_specific_qhats() docstring for more details about expected inputs.
    '''
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)

    #print(f"classwise_preds = {classwise_preds}")
    #print(f"classwise_qhats = {classwise_qhats}")
    #exit(1)

    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)

    THR = pd.DataFrame()
    for cls in np.arange(num_classes):
        all_thresholds = pd.DataFrame({'Quantile': classwise_qhats[cls], 'Label': [cls]})
        THR = pd.concat([THR, all_thresholds])

    #print(f"set_size_metrics = {set_size_metrics}, coverage_metrics = {coverage_metrics}, classwise_preds = {classwise_preds}")
    #exit(1)
    #print(f"classwise_qhats = {classwise_qhats}")

    return [np.max(classwise_qhats)], THR, classwise_preds


def get_scores_HPS(model, loader, num_classes = 10, device='cpu'):
    layer_prob = torch.nn.Softmax(dim=1)

    y_pred, logits, y_true= get_logits_targets(model, loader, num_classes, device)

    K = torch.tensor([1.0]) #Temperature scaling parameter to smooth the logits values.

    simple_outputs = layer_prob(logits/K).detach().numpy()

    return 1-simple_outputs, y_true, y_pred



########### Calculate Score ##############
def get_APS_scores(softmax_scores, labels, randomize=True, seed=0):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of APS scores
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
    
def get_APS_scores_all(softmax_scores, randomize=True, seed=0):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores
    

def get_scores_plot(model, DataLoader, num_classes = 10, device='cpu', GPU_CAPACITY=1024, all_combinations = True):
    #fx, y = get_logits_targets(model, DataLoader, BatchSize = GPU_CAPACITY, NumClass = num_classes, device = device)
    layer_prob = torch.nn.Softmax(dim=1)

    y_pred, logits, y_true= get_logits_targets(model, DataLoader, num_classes, device)

    #ratio = 0.2
    #Temp_x = logits[:int(ratio * len(y_pred))]
    #Temp_y = y_true[:int(ratio * len(y_pred))]
    #Temp_data = torch.utils.data.TensorDataset(Temp_x, Temp_y)
    #Temp_loader = torch.utils.data.DataLoader(Temp_data, batch_size = 32)

    #T = platt_logits(Temp_loader, max_iters=10, lr=0.01, epsilon=0.01)
    #T = T.detach().to('cpu')
    #print(f"T1 = {T}")
    #exit(1)
    T = torch.tensor([1.0])
    #print(f"T2 = {T}")
    #exit(1)
    #print(f"device = {logits.get_device()}, {T.get_device()}")
    #exit(1)
    simple_outputs = layer_prob(logits/T).detach().numpy()

    n = len(y_true)
    #print(f"n  = {n}")
    #exit(1)
    # create container for the scores
    #scores_simple = np.zeros((0, n, num_classes))
    rng = default_rng(seed = 0)
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)
    #print(f"shape = {simple_outputs.shape} {y_true.shape}, {n} {uniform_variables.shape}")

    scores_simple = get_APS_scores(simple_outputs, y_true)
    #print(f"shape = {scores_simple.shape}")
    return scores_simple, y_true, y_pred

def remove_empty_set(preds, val_scores):
    
    num = val_scores.shape[0]
    for i in range(num):
        max_class = np.argmax(val_scores[i])  
        preds[i].append(max_class)

    return preds
