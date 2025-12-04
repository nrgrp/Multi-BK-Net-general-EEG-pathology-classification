
import time
import os
import mne
mne.set_log_level('ERROR')

from warnings import filterwarnings
filterwarnings('ignore')

import wandb

from IPython.utils import io

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import torch
from torch.nn.functional import relu
from torch.utils.data import WeightedRandomSampler

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net,ShallowFBCSPNet,EEGNetv4, TCN
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

#from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.datautil.serialization import  load_concat_dataset

from braindecode.datasets import BaseConcatDataset
from braindecode.datautil.preprocess import preprocess, Preprocessor, exponential_moving_standardize


from braindecode.training import trial_preds_from_window_preds


from functools import partial 
from skorch.callbacks import LRScheduler, EarlyStopping,Checkpoint, EpochScoring
from skorch.helper import predefined_split


    
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score

from itertools import product
from functools import partial 

#import GPUtil
from timeit import default_timer as timer

#######################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse
import urllib
import random
import numpy as np
import optuna

import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import optuna
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from braindecode.models import get_output_shape, to_dense_prediction_model
from braindecode.datasets import BaseConcatDataset

import sys
sys.path.insert(0, '/home/code/')
from MultiBKNet import *
from ChronoNet import *


def define_model(trial,input_window_samples):
    
    df = pd.read_csv(f"/home/config_params_HPO_MultiBKNet_TUABCOMB_5fold.csv")
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_filters_conv = df["n_filters_conv"][0]
    norm_conv = df["norm_conv"][0]
    nonlin_conv = df["nonlin_conv"][0]
    pool_conv = df["pool_conv"][0]
    pool_block_conv = str(df["pool_block_conv "][0])

    fourth_block = df["fourth_block"][0]


    fourth_block_broader = df["fourth_block_broader"][0]

    drop_prob = df["drop_prob"][0]



    filter_length = df["filter_length"][0]


    pool_value = df["pool_value"][0]

    if pool_value == 'default':
        pool_time_length=3
        pool_time_stride=3

    if pool_value =='proportional':
        pool_time_length=int(filter_length/3)
        pool_time_stride=int(filter_length/3)
    

    
    
    n_chans=21
    n_classes=2
    input_window_samples=6000
    model =MultiBKNet(n_outputs= n_classes,
                 n_chans=n_chans,
                 n_filters_conv=n_filters_conv,
 
                 pool_conv=pool_conv,
                 pool_block_conv=pool_block_conv,
                 pool_conv_length=50,
                 pool_conv_stride=15,
                 norm_conv=norm_conv,
                 nonlin_conv=nonlin_conv,

                 drop_prob=drop_prob,
                 filter_length_2=filter_length,
                 filter_length_3=filter_length,
                 pool_time_length=pool_time_length,
                 pool_time_stride=pool_time_stride,

                 fourth_block=fourth_block,
                 fourth_block_broader=fourth_block_broader,


                 chs_info=None,

                 sfreq=100,

                 input_window_samples=input_window_samples,
                 add_log_softmax=True)
    

    return model #





def load_tuab(data_path):
    
    

    start = time.time()
    with io.capture_output() as captured:
        train_set = load_concat_dataset(data_path, preload=False, ids_to_load=None)

    end = time.time()


    target = train_set.description['pathological']



    for d, y in zip(train_set.datasets, target):
        d.description['pathological'] = y
        d.target_name = 'pathological'
        d.target = d.description[d.target_name]
    train_set.set_description(pd.DataFrame([d.description for d in train_set.datasets]), overwrite=True)
    

    return train_set


def get_window_tuab(input_window_samples,train_dataset, eval_dataset, eval_dataset_exb):
    

    
    
    with io.capture_output() as captured:
         window_train_set = create_fixed_length_windows(train_dataset, 
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,
                                                        preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)

    with io.capture_output() as captured:
         window_eval_set = create_fixed_length_windows(eval_dataset,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)

    with io.capture_output() as captured:
         window_eval_set_exb = create_fixed_length_windows(eval_dataset_exb,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)

    return window_train_set, window_eval_set, window_eval_set_exb


def test_model_config(model, n_chans, input_window_samples):
    try:
        out = get_output_shape(model, n_chans, input_window_samples)[2]


    except:
        out = None
        
        
    return out


def weight_function(targets, patho_factor, device='cpu'):
    # https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    #weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
    weights = torch.tensor([len(targets)/(np.count_nonzero(targets == 0)*2), (len(targets)/(np.count_nonzero(targets == 1)*2)*patho_factor)],
                        dtype=torch.float, device=device)
    return weights


def save_loss_and_model(clf, save_path, trial_number,  task_name):#trial_params,
    
    result_path = os.path.join(save_path, f"{task_name}_trial_{trial_number}_final_")
    
    results_columns = ['train_loss','valid_loss']

    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,index=clf.history[:, 'epoch'])
    df.to_csv(result_path + 'df_history.csv')
    #save history
    torch.save(clf.history, result_path + 'clf_history.py')
    

    path = result_path + "model.pt"#.format(seed)
    torch.save(clf.module, path)
    path = result_path + "state_dict.pt"#.format(seed)
    torch.save(clf.module.state_dict(), path)

    clf.save_params(f_params=result_path +'model.pkl', f_optimizer= result_path +'opt.pkl', f_history=result_path +'history.json')
    
    return df
    

    ###

def compute_and_save_recording_results(clf, window_dataset, save_path,  trial_number, task_name, seed, foldtime, cuda_memory_use,subset_name): # trial_params,
    
    all_results=list()
    
    result_path = os.path.join(save_path, f"results_{task_name}_trial_{trial_number}_final_{subset_name}")
    
    all_preds = []


    all_ys=[]

    tuh_splits = window_dataset.split([[i] for i in range(len(window_dataset.datasets))])

    for rec_i, tuh_subset in tuh_splits.items():

    
        preds_rec=clf.predict_with_window_inds_and_ys(tuh_subset)

        all_preds.append(preds_rec['preds'].mean(axis=0))
 

        all_ys.extend([int(tuh_subset.description['pathological'][0])])





    preds_per_trial= np.array(all_preds) #


    mean_preds_per_trial = np.array(all_preds)  
    y = window_dataset.description['pathological']
    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: mean_preds_per_trial[:, 0],
              column1: mean_preds_per_trial[:, 1],
              "true_pathological": y}

    assert len(y) == len(mean_preds_per_trial)

    # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_path + "_recording_predictions.csv")

    
    deep_preds =  mean_preds_per_trial[:, 0] <=  mean_preds_per_trial[:, 1]
    class_label = window_dataset.description['pathological']
    class_preds =deep_preds.astype(int)



    confusion_mat = confusion_matrix(class_label, class_preds)
    precision_normal = round((confusion_mat.T[0][0] / float(np.sum(
        confusion_mat.T[0, :]))),2)


    precision_patho = round((confusion_mat.T[1][1] / float(np.sum(
        confusion_mat.T[1, :]))),2)


    recall_normal = round((confusion_mat.T[0][0] / float(np.sum(
                confusion_mat.T[:, 0]))),2)


    recall_patho = round((confusion_mat.T[1][1] / float(np.sum(
                confusion_mat.T[:, 1]))),2)


    accuracy= round((confusion_mat.T[0][0] + confusion_mat.T[1][1])/float(np.sum(confusion_mat.T[:, :])), 2) # 

    f1_score_patho = round((2 * precision_patho * recall_patho / (precision_patho + recall_patho)),2)
    f1_score_patho

    f1_score_normal = round((2 * precision_normal * recall_normal / (precision_normal + recall_normal)),2)
    f1_score_normal
    
    tn, fp, fn, tp = confusion_matrix(class_label, class_preds).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    acc = accuracy_score(class_label, class_preds)
    
    bal_acc = balanced_accuracy_score(class_label, class_preds)
    
    roc_auc = roc_auc_score(class_label, class_preds)
    
    pr_auc= average_precision_score(class_label, class_preds)

    f1_score_normal = f1_score(class_label, class_preds, pos_label=0)
    f1_score_patho = f1_score(class_label, class_preds, pos_label=1)
    
    precision_score_normal = precision_score(class_label, class_preds, pos_label=0)
    precision_score_patho = precision_score(class_label, class_preds, pos_label=1)
    
    recall_score_normal = recall_score(class_label, class_preds, pos_label=0)
    recall_score_patho = recall_score(class_label, class_preds, pos_label=1)
    
    
    results = {
        'trial_number':trial_number,
        'task_name': task_name,
        'subset_name': subset_name,
      
        'seed': seed,
        'num_rec abnormal' : len(window_dataset.description[window_dataset.description['pathological']==1]),
        'num_rec normal' : len(window_dataset.description[window_dataset.description['pathological']==0]),
        'precision_normal': precision_normal*100,
        'precision_patho': precision_patho*100,
        'recall_normal': recall_normal*100,
        'recall_patho': recall_patho*100,
        'f1_score_patho1': f1_score_patho*100,
        'f1_score_normal1': f1_score_normal*100,
        'accuracy': accuracy*100,
        'accuracy2': acc,
        'balanced_accuracy': bal_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score_patho':f1_score_patho,
        'f1_score_normal': f1_score_normal,
        'precision_score_patho': precision_score_patho,
        'precision_score_normal': precision_score_normal,
        'recall_score_patho': recall_score_patho,
        'recall_score_normal': recall_score_normal,
        'TN':tn,
        'FN': fn,
        'TP':tp,
        'FP': fp,            
        'cuda_memory_use': cuda_memory_use,# (torch.cuda.max_memory_allocated() / 1e9)
        'time_used_in_s': foldtime,
        
    }


    
    all_results.append(results)
    results_df = pd.DataFrame(all_results)
    fname = result_path +'_recording_results.csv'
    results_df.to_csv(fname)

    sensitivity= recall_patho*100
    specificity= recall_normal*100
    accuracy= accuracy*100
    




def compute_and_save_window_results(clf, window_dataset, save_path,  trial_number,  task_name, seed, foldtime, cuda_memory_use,subset_name):  #trial_params,
    
    all_results=list()
    
    result_path = os.path.join(save_path, f"results_{task_name}_trial_{trial_number}_seed_{seed}_final_{subset_name}")

    win_preds =[]

    win_ys = []

    win_rec_number = []

    trial_preds=[]

    tuh_splits = window_dataset.split([[i] for i in range(len(window_dataset.datasets))])

    for rec_i, tuh_subset in tuh_splits.items():

        preds=clf.predict_with_window_inds_and_ys(tuh_subset)

        win_preds.extend(preds['preds'])

        win_ys.extend(preds['window_ys'])

        win_rec_number.extend([int(rec_i)]*preds['preds'].shape[0])


    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: np.array(win_preds)[:, 0],
                  column1: np.array(win_preds)[:, 1],
                  "true_pathological": np.array(win_ys),
                 "rec_i":np.array(win_rec_number)}
        # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_path +"prediction_windows.csv")


    deep_preds_win =  np.array(np.array(win_preds)[:, 0] <= np.array(win_preds)[:, 1]) 
    class_label_win  = np.array(win_ys)
    class_preds_win  =deep_preds_win.astype(int)


    confusion_mat_win = confusion_matrix(class_label_win, class_preds_win)
    precision_normal_win = round((confusion_mat_win.T[0][0] / float(np.sum(
        confusion_mat_win.T[0, :]))),2)


    precision_patho_win = round((confusion_mat_win.T[1][1] / float(np.sum(
        confusion_mat_win.T[1, :]))),2)


    recall_normal_win = round((confusion_mat_win.T[0][0] / float(np.sum(
                confusion_mat_win.T[:, 0]))),2)


    recall_patho_win = round((confusion_mat_win.T[1][1] / float(np.sum(
                confusion_mat_win.T[:, 1]))),2)


    accuracy_win= round((confusion_mat_win.T[0][0] + confusion_mat_win.T[1][1])/float(np.sum(confusion_mat_win.T[:, :])), 2) # overall_accuracy



    f1_score_patho_win = round((2 * precision_patho_win * recall_patho_win / (precision_patho_win + recall_patho_win)),2)
    f1_score_patho_win

    f1_score_normal_win = round((2 * precision_normal_win * recall_normal_win / (precision_normal_win + recall_normal_win)),2)
    f1_score_normal_win

 
    tn, fp, fn, tp = confusion_matrix(class_label_win, class_preds_win).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    acc = accuracy_score(class_label_win, class_preds_win)
    
    bal_acc = balanced_accuracy_score(class_label_win, class_preds_win)
    
    roc_auc = roc_auc_score(class_label_win, class_preds_win)
    
    pr_auc= average_precision_score(class_label_win, class_preds_win)

    f1_score_normal = f1_score(class_label_win, class_preds_win, pos_label=0)
    f1_score_patho = f1_score(class_label_win, class_preds_win, pos_label=1)
    
    precision_score_normal = precision_score(class_label_win, class_preds_win, pos_label=0)
    precision_score_patho = precision_score(class_label_win, class_preds_win, pos_label=1)
    
    recall_score_normal = recall_score(class_label_win, class_preds_win, pos_label=0)
    recall_score_patho = recall_score(class_label_win, class_preds_win, pos_label=1)

    
    
    results = {
        'trial_number':trial_number,
        'task_name': task_name,
        'subset_name': subset_name,
    
        'seed':seed,
        'num_rec abnormal' : len(window_dataset.description[window_dataset.description['pathological']==1]),
        'num_rec normal' : len(window_dataset.description[window_dataset.description['pathological']==0]),

        'precision_normal': precision_normal_win*100,
        'precision_patho': precision_patho_win*100,
        'recall_normal': recall_normal_win*100,
        'recall_patho': recall_patho_win*100,
        'f1_score_patho1': f1_score_patho_win*100,
        'f1_score_normal1': f1_score_normal_win*100,
        'accuracy': accuracy_win*100,
        'accuracy2': acc,
        'balanced_accuracy': bal_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score_patho':f1_score_patho,
        'f1_score_normal': f1_score_normal,
        'precision_score_patho': precision_score_patho,
        'precision_score_normal': precision_score_normal,
        'recall_score_patho': recall_score_patho,
        'recall_score_normal': recall_score_normal,
        'TN':tn,
        'FN': fn,
        'TP':tp,
        'FP': fp,          
        
    }
    

    all_results.append(results)
    results_df = pd.DataFrame(all_results)
    fname = result_path +'_window_results.csv'
    results_df.to_csv(fname)

    
    sensitivity_win= recall_patho_win*100
    specificity_win= recall_normal_win*100
    accuracy_win=accuracy_win*100
    





def objective(trial, seed,train_dataset, eval_dataset,eval_dataset_exb,save_path,task_name): #optuna.Trial) -> float: #trial):
    
        
    n_chans=21
    n_classes=2
    input_window_samples = 6000
    # Generate the model.
    set_random_seeds(seed=seed, cuda=True)
    model = define_model(trial,input_window_samples)
    
    model_size = count_parameters(model)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
#         set_random_seeds(seed=seed, cuda=True)
        model.cuda(device)

    df = pd.read_csv(f"/home/config_params_HPO_MultiBKNet_TUABCOMB_5fold.csv")
    
    
    patho_factor= int(df['patho_factor_weighted_loss'][0])
    
    optimizer_name = "AdamW"
    
    beta1 = df['beta1'][0]
    
   # betas = 
    lr = df['lr'][0]
    weight_decay = df['weight_decay'][0]
    optimizer = getattr(optim, optimizer_name)  

    batch_size = int(df['batch_size'][0])

    n_epochs =int(df['n_epochs'][0]) 
    
    trial_number = trial



    window_train_set, window_eval_set, window_eval_set_exb = get_window_tuab(input_window_samples,train_dataset, eval_dataset, eval_dataset_exb)

    clf = EEGClassifier(model,cropped=False,
                        criterion=torch.nn.NLLLoss(),
                        criterion__weight=weight_function(window_train_set.get_metadata().target,patho_factor, device),
                        optimizer=optimizer,#
                        train_split=predefined_split(window_eval_set),
                   
                        optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay,
                        optimizer__betas=(beta1, 0.999),
                        iterator_train__shuffle=True,
                        iterator_train__num_workers=2,
                        batch_size=batch_size,
              
                        callbacks=[("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),],  
                        device=device)
    start = timer()

    clf.fit(window_train_set, y=None, epochs=n_epochs)
    
    end = timer()
    
    foldtime= end - start

    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    
    cuda_memory_use =(torch.cuda.max_memory_allocated() / 1e9)

 

    df_loss = save_loss_and_model(clf, save_path, trial_number,  task_name)#

    train_loss = df_loss['train_loss'].iloc[-1]
    valid_loss = df_loss['valid_loss'].iloc[-1]

    compute_and_save_recording_results(clf, window_eval_set, save_path,  trial_number, task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuab')

    compute_and_save_window_results(clf, window_eval_set, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuab')
    
    compute_and_save_recording_results(clf, window_eval_set_exb, save_path,  trial_number, task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuabexb')

    compute_and_save_window_results(clf, window_eval_set_exb, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuabexb')

    compute_and_save_recording_results(clf, window_train_set, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='train')

    compute_and_save_window_results(clf, window_train_set, save_path,  trial_number,  task_name,seed,foldtime,cuda_memory_use,subset_name='train')





def objective_cv(trial, seed,save_path,task_name):
    
    train_folder = '/home/TUABCOMB/final_train/'
    
    test_folder = '/home/TUAB/final_eval/'
    test_folder_exb= '/home/TUABEXB/final_eval/'
    #
    df = pd.read_csv('/home/TUABCOMB_trainset_5fold.csv')

  
    dataset= load_tuab(train_folder)
    
    trial_number=trial#.number
    

    save_path_trial = save_path 
    
    if not os.path.exists(save_path_trial):
        os.makedirs(save_path_trial)
    
    

    train_dataset = load_tuab(train_folder) 

    eval_dataset = load_tuab(test_folder)
    
    eval_dataset_exb = load_tuab(test_folder_exb)

    start_fold = timer()

    objective(trial, seed,train_dataset, eval_dataset,eval_dataset_exb,save_path=save_path_trial,task_name=task_name)

    end_fold =  timer()



    fold_time = end_fold -start_fold

