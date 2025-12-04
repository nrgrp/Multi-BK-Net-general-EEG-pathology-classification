import time
import os
import mne
mne.set_log_level('ERROR')

from warnings import filterwarnings
filterwarnings('ignore')

import wandb
os.environ["WANDB_API_KEY"] = 'xxxxxxxxx'
#os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_CACHE_DIR"] = '/home/wandb/'
os.environ["WANDB_CONFIG_DIR"] = '/home/wandb/'
os.environ["WANDB_DIR"] = '/home/wandb/'

from IPython.utils import io

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import torch
from torch.nn.functional import relu, elu, gelu
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
#from optuna.integration import SkorchPruningCallback
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



def define_model(trial,input_window_samples):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_filters_conv = trial.suggest_categorical("n_filters_conv", [20,25,30,35,40,45,50,55,60,65,70,75,80])
    norm_conv = trial.suggest_categorical('norm_conv', ['batch', 'group'])
    nonlin_conv = trial.suggest_categorical('nonlin_conv', ['elu', 'gelu'])
    pool_conv = trial.suggest_categorical('pool_conv', ['mean', 'max'])
    pool_block_conv = trial.suggest_categorical('pool_block_conv ', ['mean', 'max'])

    fourth_block = trial.suggest_categorical('fourth_block', [True, False])


    fourth_block_broader = trial.suggest_categorical('fourth_block_broader', [True, False])

    drop_prob = trial.suggest_float("drop_prob", 0.4, 0.6)



    filter_length = trial.suggest_categorical("filter_length", [10,15,20])


    pool_value = trial.suggest_categorical("pool_value", ['default', 'proportional'])

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
        train_set = load_concat_dataset(data_path, preload=False, ids_to_load=None)#

    end = time.time()



    target = train_set.description['pathological']



    for d, y in zip(train_set.datasets, target):
        d.description['pathological'] = y
        d.target_name = 'pathological'
        d.target = d.description[d.target_name]
    train_set.set_description(pd.DataFrame([d.description for d in train_set.datasets]), overwrite=True)
    

    return train_set


def get_window_tuab(input_window_samples,train_dataset, valid_dataset):
    

    
    
    with io.capture_output() as captured:
         window_train_set = create_fixed_length_windows(train_dataset, 
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,
                                                        preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)

    with io.capture_output() as captured:
         window_valid_set = create_fixed_length_windows(valid_dataset,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)



    return window_train_set, window_valid_set#


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

def save_loss_and_model(clf, save_path,ifold, trial_number, trial_params, task_name):
    
    result_path = os.path.join(save_path, f"{task_name}_trial_{trial_number}_fold_{ifold}_")
    
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

def compute_and_save_recording_results(clf, window_dataset, save_path, ifold, trial_number, trial_params, task_name, seed, subset_name):
    
    all_results=list()
    
    result_path = os.path.join(save_path, f"results_{task_name}_trial_{trial_number}_fold_{ifold}_{subset_name}")
    
    all_preds = []


    all_ys=[]

    tuh_splits = window_dataset.split([[i] for i in range(len(window_dataset.datasets))])

    for rec_i, tuh_subset in tuh_splits.items():


        preds_rec=clf.predict_with_window_inds_and_ys(tuh_subset)

        all_preds.append(preds_rec['preds'].mean(axis=0))
       # all_preds.extend(np.mean(preds_trial[0])

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


    accuracy= round((confusion_mat.T[0][0] + confusion_mat.T[1][1])/float(np.sum(confusion_mat.T[:, :])), 2) # overall_accuracy



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
        'ifold': ifold,
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
        
    }


    results.update(trial_params)
    
    all_results.append(results)
    results_df = pd.DataFrame(all_results)
    fname = result_path +'_recording_results.csv'
    results_df.to_csv(fname)
    #print(f'Results saved under {fname}.')
    
    sensitivity= recall_patho*100
    specificity= recall_normal*100
    accuracy= accuracy*100
    
    return accuracy, sensitivity, specificity



def compute_and_save_window_results(clf, window_dataset, save_path, ifold, trial_number, trial_params, task_name, seed, subset_name):
    
    all_results=list()
    
    result_path = os.path.join(save_path, f"results_{task_name}_trial_{trial_number}_fold_{ifold}_{subset_name}")
    
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


    deep_preds_win =  np.array(np.array(win_preds)[:, 0] <= np.array(win_preds)[:, 1]) #
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
        'ifold': ifold,
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
    

    results.update(trial_params)
    
    all_results.append(results)
    results_df = pd.DataFrame(all_results)
    fname = result_path +'_window_results.csv'
    results_df.to_csv(fname)
    #print(f'Results saved under {fname}.')
    
    sensitivity_win= recall_patho_win*100
    specificity_win= recall_normal_win*100
    accuracy_win=accuracy_win*100
    
    return accuracy_win, sensitivity_win, specificity_win




def objective(trial,ifold, seed,train_dataset, valid_dataset,save_path,task_name): 
    
        
    n_chans=21
    n_classes=2
    input_window_samples = trial.suggest_categorical("input_window_samples", [6000])
    # Generate the model.
    set_random_seeds(seed=seed, cuda=True)
    model = define_model(trial,input_window_samples)
    
    model_size = count_parameters(model)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

        model.cuda(device)

    patho_factor= trial.suggest_categorical("patho_factor_weighted_loss", [1,4])
    
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
    
    beta1 = trial.suggest_categorical("beta1", [0.5,0.9])
    
   # betas = 
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)  

    batch_size = trial.suggest_categorical("batch_size", [64,32,16])

    n_epochs = trial.suggest_int("n_epochs", 30,70) 



    config = dict(trial.params)
    config["trial.number"] = trial.number
    config["fold"] = ifold
    config["model_size"] = model_size

    pd.DataFrame(dict(trial.params),index=[0]).to_csv(os.path.join(f"/home/{STUDY_NAME}/",f"trial_{trial.number}_params_{task_name}.csv"))
    pd.DataFrame(config,index=[0]).to_csv(os.path.join(f"/home/{STUDY_NAME}/",f"config_trial_{trial.number}_params_{task_name}.csv"))
    trial_number =trial.number
    trial_params =trial.params
    
    print('trial_number: ', trial_number)

    wanddir='/home/wandb/'
    wandb_fold= wandb.init(
        project=task_name,#"
        config=config,
        name=f"trial_number_{trial_number}_fold_{ifold}",
        group=f"trial_number_{trial_number}",
        reinit=True,
        dir=wanddir,
        settings=wandb.Settings(start_method="thread")
    )



    window_train_set, window_valid_set = get_window_tuab(input_window_samples,train_dataset, valid_dataset)

    clf = EEGClassifier(model,cropped=False,
                        criterion=torch.nn.NLLLoss(),
                        criterion__weight=weight_function(window_train_set.get_metadata().target,patho_factor, device),
                        optimizer=optimizer,
                        train_split=predefined_split(window_valid_set),
          
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
    
    config["cuda_memory_used"]=(torch.cuda.max_memory_allocated() / 1e9)
    
    
    
    config["time_used_in_s"]=foldtime

    pd.DataFrame(config,index=[0]).to_csv(os.path.join(f"/home/{STUDY_NAME}/",f"config_trial_{trial.number}_fold_{ifold}_params_{task_name}.csv"))

    df_loss = save_loss_and_model(clf, save_path,ifold, trial_number, trial_params, task_name)

    train_loss = df_loss['train_loss'].iloc[-1]
    valid_loss = df_loss['valid_loss'].iloc[-1]

    val_accuracy, val_sensitivity, val_specificity = compute_and_save_recording_results(clf, window_valid_set, save_path, ifold, trial_number, trial_params, task_name, seed,subset_name='validation')



    train_accuracy, train_sensitivity, train_specificity = compute_and_save_recording_results(clf, window_train_set, save_path, ifold, trial_number, trial_params, task_name, seed,subset_name='train')



    wandb_fold.define_metric("validation accuracy", summary="max",step_metric="fold")
    wandb_fold.define_metric("validation sensitivity", summary="max",step_metric="fold")

    wandb_fold.log(data={"validation accuracy": val_accuracy, "validation sensitivity": val_sensitivity,"validation specificity": val_specificity, "validation_loss":valid_loss,
                        "train accuracy": val_accuracy, "train sensitivity": val_sensitivity,"train specificity": val_specificity, "train_loss":train_loss, "cuda_memory_used": (torch.cuda.max_memory_allocated() / 1e9),
                        "time_used_in_s": foldtime, "model_size": model_size})


    wandb_fold.finish(quiet=True)

    return val_accuracy, val_sensitivity, val_specificity


def objective_cv(trial, seed,save_path,task_name):
    
    train_folder = '/home/TUABCOMB/final_train/'  
    #
    df = pd.read_csv('/home/TUABCOMB_trainset_5fold.csv')
    

    scores_acc = []
    scores_sens =[]
    scores_spec=[]
    dataset= load_tuab(train_folder)
    
    trial_number=trial.number
    

    
    save_path_trial = os.path.join(save_path, task_name, str(trial_number))
    
    if not os.path.exists(save_path_trial):
        os.makedirs(save_path_trial)
    
    
    config = dict(trial.params)
    config["trial.number"] = trial.number
    
    print(trial.params)
    
    
    wanddir='/home/wandb/'
    
   
    for ifold in range(1,6): #####CHANGE
        
        valid_fold = ifold
        
    # split training data into development set and validation set
        valid_ind = list(df[df['kfold']==valid_fold]['original_id'])

        # Define ind for development fold
        # select all recordings of the remaining folds as training fold
        # Thus at the same recordings with kfold = NaN, i.e. no clinical history in report, are excluded
        trainfolds = list(range(1,11)) # here 10-fold cv is used
        trainfolds.remove(valid_fold)
      
        mask = df['kfold'].isin(trainfolds)
        train_ind = list(df[mask]['original_id'])#[:50] ####CHANGE
        
        
        train_dataset = dataset.split(train_ind)['0']

        valid_dataset = dataset.split(valid_ind)['0']
        
        start_fold = timer()
        
        accuracy, sensitivity, specificity = objective(trial,ifold, seed,train_dataset, valid_dataset,save_path=save_path_trial,task_name=task_name)
        
        end_fold =  timer()
        
        scores_acc.append(accuracy)
        scores_sens.append(sensitivity)
        scores_spec.append(specificity)
        

        
        print(accuracy)
        
        fold_time = end_fold -start_fold
        
        if ifold==1:
            if accuracy <= 75.0 or sensitivity <= 75.0 or fold_time >  162000: #
                break

    
    val_accuracy = np.mean(scores_acc)#*100
    val_sensitivity = np.mean(scores_sens)
    val_specificity = np.mean(scores_spec)

    
    print(val_accuracy)


    #config["fold"] = ifold

    wandb.init(
            project=task_name,#"optuna",
        #    entity="nzw0301",  # NOTE: this entity depends on your wandb account.
            config=config,
            name=f"trial_number_{trial_number}",
            group="all_trials",#f"trial_number_{trial_number}",  
            reinit=True,
            dir=wanddir,
            settings=wandb.Settings(start_method="thread")
        )
    
    if ifold==1:
        if val_accuracy <=75.0 or val_sensitivity <= 75.0 or fold_time >  162000:#
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()
        
        
    wandb.define_metric("validation accuracy", summary="best",step_metric="trial")
    wandb.define_metric("validation sensitivity", summary="best",step_metric="trial")

    
    wandb.log(data={"validation accuracy": val_accuracy, "validation sensitivity": val_sensitivity,"validation specificity": val_specificity})
    

        
    
    wandb.run.summary["accuracy"] = val_accuracy
    wandb.run.summary["state"] = "complated"
   
    
    
    return val_accuracy, val_sensitivity




if __name__ == "__main__":
    
    from random import randint
    from time import sleep

    random_sec = randint(10,60)
    print("start in", random_sec)
    
    
    sleep(random_sec)
    
    seed=2010629
   # n_epochs=2

    wandb.login()

    STUDY_NAME = "HPO_MultiBKNet_TUABCOMB_5fold"  #multivariate
    
    rdb_string_url = "sqlite:///" + os.path.join('/home/HPO/', (STUDY_NAME  + ".db"))
    rdb_raw_bytes_url = r'{}'.format(rdb_string_url)##see https://stackoverflow.com/questions/56416437/confusion-about-uri-path-to-configure-sqlite-database
    


    storage_instance = optuna.storages.RDBStorage(url=rdb_raw_bytes_url,heartbeat_interval=60, grace_period=180)  # Grace period before a running trial is failed from the last heartbeat. here 3min
    
   # storage = optuna.storages.RDBStorage(url="sqlite:///:memory:", heartbeat_interval=60, grace_period=120)
        
    save_path = '.g/HPO/'#'
    
    task_name= STUDY_NAME#'test_optuna'
    
    wandb_kwargs = {
            "project": STUDY_NAME,#"try_optuna",
          #  "entity": "nzw0301",
            "config": {"sampler": optuna.samplers.TPESampler.__name__},
            "reinit": True,
             "dir": '/home/wandb/'}

    wandbc = WeightsAndBiasesCallback(
            metric_name="validation accuracy", wandb_kwargs=wandb_kwargs)#
       
    # Recording heartbeats every 60 seconds.
    # Other processes' trials where more than 120 seconds have passed
    # since the last heartbeat was recorded will be automatically failed.
   # storage = optuna.storages.RDBStorage(url="sqlite:///:memory:", heartbeat_interval=60, grace_period=180)
    #study = optuna.create_study()
    
    #storage_instance = optuna.storages.RDBStorage(url=rdb_raw_bytes_url,heartbeat_interval=60, grace_period=180)
    
    study = optuna.create_study(directions=["maximize","maximize"], study_name=STUDY_NAME, sampler=optuna.samplers.TPESampler(multivariate=True), storage=storage_instance, load_if_exists=True) #storage=rdb_raw_bytes_url
   


    df = study.trials_dataframe(attrs=('number', 'value',  'params','state') )
    
    
   
    result_path = os.path.join(save_path, task_name)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    df.to_csv(os.path.join(result_path, f"trials_dataframe_{task_name}.csv"))
    df.to_csv(os.path.join(f"/home/{STUDY_NAME}/",f"trials_dataframe_{task_name}.csv"))
    
    
    func = lambda trial: objective_cv(trial, seed,save_path,task_name)  #see https://www.kaggle.com/discussions/general/261870
    
    study.optimize(func, n_trials=1,callbacks=[wandbc])
  
    
    df = study.trials_dataframe(attrs=('number', 'value',  'params','state') )
    
    
   
    result_path = os.path.join(save_path, task_name)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    df.to_csv(os.path.join(result_path, f"trials_dataframe_{task_name}.csv"))
    df.to_csv(os.path.join(f"/home/{STUDY_NAME}/",f"trials_dataframe_{task_name}.csv"))
 

    wandb.run.summary["best accuracy"] = study.best_trial.value

    wandb.log({"optuna_optimization_history": optuna.visualization.plot_optimization_history(study),
            "optuna_param_importances": optuna.visualization.plot_param_importances(study),})

    wandb.finish()
    


