import time
import os
import mne
mne.set_log_level('ERROR')

from warnings import filterwarnings
filterwarnings('ignore')


from IPython.utils import io

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from timeit import default_timer as timer
import torch
from torch.nn.functional import relu
from torch.utils.data import WeightedRandomSampler

from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net,ShallowFBCSPNet,EEGNetv4, TCN
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape

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

import sys
sys.path.insert(0, './code/')
from ChronoNet import *

from itertools import product
from functools import partial 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def plot_confusion_matrix_paper(confusion_mat, p_val_vs_a=None,
                                p_val_vs_b=None,
                                class_names=None, figsize=None,
                                colormap=cm.bwr,
                                textcolor='black', vmin=None, vmax=None,
                                fontweight='normal',
                                rotate_row_labels=90,
                                rotate_col_labels=0,
                                with_f1_score=False,
                                norm_axes=(0, 1),
                                rotate_precision=False,
                                class_names_fontsize=12):
    
    # TODELAY: split into several functions
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    # then have to transpose pvals also
    if p_val_vs_a is not None:
        p_val_vs_a = p_val_vs_a.T
    if p_val_vs_b is not None:
        p_val_vs_b = p_val_vs_b.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in range(n_classes)]

    # norm by number of targets (targets are columns after transpose!)
    # normed_conf_mat = confusion_mat / np.sum(confusion_mat,
    #    axis=0).astype(float)
    # norm by all targets
    normed_conf_mat = confusion_mat / np.float32(np.sum(confusion_mat, axis=norm_axes, keepdims=True))

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if vmin is None:
        vmin = np.min(normed_conf_mat)
    if vmax is None:
        vmax = np.max(normed_conf_mat)

    # see http://stackoverflow.com/a/31397438/1469195
    # brighten so that black text remains readable
    # used alpha=0.6 before
    def _brighten(x, ):
        brightened_x = 1 - ((1 - np.array(x)) * 0.4)
        return brightened_x

    brightened_cmap = _cmap_map(_brighten, colormap) #colormap #
    ax.imshow(np.array(normed_conf_mat), cmap=brightened_cmap,
              interpolation='nearest', vmin=vmin, vmax=vmax)

    # make space for precision and sensitivity
    plt.xlim(-0.5, normed_conf_mat.shape[0]+0.5)
    plt.ylim(normed_conf_mat.shape[1] + 0.5, -0.5)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in range(width):
        for y in range(height):
            if x == y:
                this_font_weight = 'bold'
            else:
                this_font_weight = fontweight
            annotate_str = "{:d}".format(confusion_mat[x][y])
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
                annotate_str += " *"
            else:
                annotate_str += "  "
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
                annotate_str += u"*"
            if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
                annotate_str += u"*"

            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
                annotate_str += u" ◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
                annotate_str += u"◊"
            if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
                annotate_str += u"◊"
            annotate_str += "\n"
            ax.annotate(annotate_str.format(confusion_mat[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color=textcolor,
                        fontweight=this_font_weight)
            if x != y or (not with_f1_score):
                ax.annotate(
                    "\n\n{:4.1f}%".format(
                        normed_conf_mat[x][y] * 100),
                    #(confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)
            else:
                assert x == y
                precision = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[x, :]))
                sensitivity = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[:, y]))
                f1_score = 2 * precision * sensitivity / (precision + sensitivity)

                ax.annotate("\n{:4.1f}%\n{:4.1f}% (F)".format(
                    (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100,
                    f1_score * 100),
                    xy=(y, x + 0.1),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)

    # Add values for target correctness etc.
    for x in range(width):
        y = len(confusion_mat)
        correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    for y in range(height):
        x = len(confusion_mat)
        correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
        annotate_str = ""
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.05:
            annotate_str += " *"
        else:
            annotate_str += "  "
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.01:
            annotate_str += u"*"
        if p_val_vs_a is not None and p_val_vs_a[x][y] < 0.001:
            annotate_str += u"*"

        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.05:
            annotate_str += u" ◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.01:
            annotate_str += u"◊"
        if p_val_vs_b is not None and p_val_vs_b[x][y] < 0.001:
            annotate_str += u"◊"
        annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                xy=(len(confusion_mat), len(confusion_mat)),
                horizontalalignment='center',
                verticalalignment='center', fontsize=12,
                fontweight='bold')

    plt.xticks(range(width), class_names, fontsize=class_names_fontsize,
               rotation=rotate_col_labels)
    plt.yticks(np.arange(0,height), class_names,
               va='center',
               fontsize=class_names_fontsize, rotation=rotate_row_labels)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)

    # n classes is also shape of matrix/size
    ax.text(-1.2, n_classes+0.2, "Specificity /\nSensitivity", ha='center', va='center',
            fontsize=13)
    if rotate_precision:
        rotation=90
        x_pos = -1.1
        va = 'center'
    else:
        rotation=0
        x_pos = -0.8
        va = 'top'
    ax.text(n_classes, x_pos, "Precision", ha='center', va=va,
            rotation=rotation,  # 270,
            fontsize=13)

    return fig

# see http://stackoverflow.com/a/31397438/1469195
def _cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    from matplotlib.colors import LinearSegmentedColormap as lsc
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: list(map(lambda x: x[0], cdict[key])) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_dicts = np.array(list(step_dict.values()))
    step_list = np.unique(step_dicts)
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(list(map(function, y0)))
    y1 = np.array(list(map(function, y1)))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    # Remove alpha, otherwise crashes...
    step_dict.pop('alpha', None)
    return lsc(name, step_dict, N=N, gamma=gamma)

###########################
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



def plot_roc_curve(fper, tper, save_name):
    plt.clf()
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(save_name +'_ROC.png',bbox_inches='tight')
    plt.show()
    
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


def get_window_tuab(input_window_samples,train_dataset, eval_dataset, eval_dataset2):
    

    
    
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
         window_eval_set2 = create_fixed_length_windows(eval_dataset2,
                                                        start_offset_samples=0,
                                                        stop_offset_samples=None,preload=False,
                                                        window_size_samples=input_window_samples,
                                                        window_stride_samples=input_window_samples,
                                                        drop_last_window=False,)

    return window_train_set, window_eval_set,window_eval_set2#

   

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
    
  ####################################

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
   # print(table)
  #  print(f"Total Trainable Params: {total_params}")
    return total_params


def train_and_eval_TUHEEG_pathology(model_name,
                     drop_prob,
                     batch_size,
                     lr,
                     n_epochs,
                     weight_decay,
                     result_folder,
                     train_folder,
                     eval_folder,
                     eval_folder2,
                     task_name,
                     ids_to_load_train,
                     stride_size,
                     input_window_samples,
                    # ids_to_load_valid,
                    # ifold,
                     seed,
                     cuda = True,
                     sfreq=100,
                          ):

    ###################################
    target_name = None  # Comment Lukas our target is not supported by default, set None and add later 
    add_physician_reports = True

    n_minutes = 20 #

    n_max_minutes = n_minutes+1
    

    N_REPETITIONS =1
    ####### MODEL DEFINITION ############
    torch.backends.cudnn.benchmark = True
    ######


    set_random_seeds(seed=seed, cuda=cuda)

    

    print(task_name)
   # print(ifold)

    print('loading data')
    # load from train
    train_dataset = load_tuab(train_folder) #dataset.split(train_ind)['0']

    eval_dataset = load_tuab(eval_folder)
    
    eval_dataset2 = load_tuab(eval_folder2)

    window_train_set, window_eval_set, window_eval_set_exb = get_window_tuab(input_window_samples,train_dataset, eval_dataset, eval_dataset2)
            

    #del  train_set
    print('abnormal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==1])))
    print('normal train ' + str(len(window_train_set.description[window_train_set.description['pathological']==0])))

  
    
    print('abnormal eval' + str(len(window_eval_set.description[window_eval_set.description['pathological']==1])))


    print('normal eval ' + str(len(window_eval_set.description[window_eval_set.description['pathological']==0])))
   # all_results = list()

   
    if not os.path.exists(result_folder + model_name + '/' + 'seed'+str(seed)):
            os.mkdir(result_folder + model_name + '/' + 'seed'+str(seed))
            print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " Created ")
    else:    
        print("Directory " , result_folder + model_name + '/' + 'seed'+str(seed) ,  " already exists")
            

    dirSave = result_folder + model_name + '/' + 'seed'+str(seed) + '/' + task_name    
    save_name =model_name   +'_trained_' +  str(task_name)+ '_'  
   # cv = 'SKF'    
    if not os.path.exists(dirSave):
        os.mkdir(dirSave)
        print("Directory " , dirSave ,  " Created ")
    else:    
        print("Directory " , dirSave ,  " already exists")




    result_path = dirSave  +'/' + save_name


                        # for pathology decoding
    n_classes=2
    # Extract number of chans from dataset
    n_chans = 21

    if model_name == 'Deep4Net':
        n_start_chans = 25
        final_conv_length = 'auto'
        n_chan_factor = 2
        stride_before_pool = True
       # input_window_samples =6000
        model = Deep4Net(
                    n_chans, n_classes,
                    n_filters_time=n_start_chans,
                    n_filters_spat=n_start_chans,
                    input_window_samples=input_window_samples,
                    n_filters_2=int(n_start_chans * n_chan_factor),
                    n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                    n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                    final_conv_length=final_conv_length,
                    stride_before_pool=stride_before_pool,

                    drop_prob=drop_prob)
            # Send model to GPU
            

        if cuda:
            model.cuda()
            
            
    elif model_name == 'Shallow':  
        n_start_chans = 40
        final_conv_length = 'auto'#25
        #input_window_samples =6000
        model = ShallowFBCSPNet(n_chans,n_classes,
                                input_window_samples=input_window_samples,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                final_conv_length= final_conv_length,
                                drop_prob=drop_prob)
        # Send model to GPU
        if cuda:
            model.cuda()

#
     
    elif model_name == 'TCN':  
        n_chan_factor = 2
        stride_before_pool = True

        l2_decay = 1.7491630095065614e-08

        gradient_clip = 0.25

        model = TCN(
            n_in_chans=n_chans, n_outputs=n_classes,
            n_filters=55,
            n_blocks=5,
            kernel_size=16,
            drop_prob=drop_prob,
            add_log_softmax=True,

            )
        model = torch.nn.Sequential(model, torch.nn.AdaptiveAvgPool1d(1),  torch.nn.Flatten() )
            # Send model to GPU
        if cuda:
            model.cuda()

    elif model_name == 'EEGNet':    

      #  input_window_samples =6000
        final_conv_length='auto'#18
        drop_prob=0.25
        #lr = 0.001
       # weight_decay =  0
        model = EEGNetv4(
            n_chans, n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=final_conv_length,
            drop_prob=drop_prob)
        if cuda:
            model.cuda()
     
 
    elif model_name =='ChronoNet':
        
        
        
    
   
    print(model_name + ' model sent to cuda')
    ###GPUtil.showUtilization()





    if model_name =='ChronoNet':
        
        clf = EEGClassifier(model,cropped=False,
                    criterion=torch.nn.CrossEntropyLoss, #
                    optimizer=torch.optim.Adam,
                    train_split=predefined_split(window_eval_set),
                    optimizer__lr=lr,

                    iterator_train__shuffle=True,
                    iterator_train__num_workers=2,
                    batch_size=batch_size,
                    callbacks=[("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),],  #
                    device='cuda')
            
    else:
        clf = EEGClassifier(model,cropped=False,
                        criterion=torch.nn.NLLLoss, 
                        optimizer=torch.optim.AdamW,
                        train_split=predefined_split(window_eval_set),
                        optimizer__lr=lr,
                        optimizer__weight_decay=weight_decay,
                        iterator_train__shuffle=True,
                        iterator_train__num_workers=2,
                        batch_size=batch_size,
                        callbacks=[("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),],  #
                        device='cuda')
    



    print('classifier initialized')
    start = timer()
    clf.fit(window_train_set, y=None, epochs=n_epochs)
    end = timer()
    foldtime= end - start

    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    used_memory=(torch.cuda.max_memory_allocated() / 1e9)
    print('end training')
          #


#                 ###################################
    ####### SAVE ############
    print('saving')
    # save model
    df_loss = save_loss_and_model(clf, save_path, trial_number,  task_name)
    
    compute_and_save_recording_results(clf, window_eval_set, save_path,  trial_number, task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuab')

    compute_and_save_window_results(clf, window_eval_set, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuab')

    compute_and_save_recording_results(clf, window_eval_set_exb, save_path,  trial_number, task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuabexb')

    compute_and_save_window_results(clf, window_eval_set_exb, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='eval_tuabexb')
    
    compute_and_save_recording_results(clf, window_train_set, save_path,  trial_number,  task_name, seed,foldtime,cuda_memory_use,subset_name='train')

    compute_and_save_window_results(clf, window_train_set, save_path,  trial_number,  task_name,seed,foldtime,cuda_memory_use,subset_name='train')

    
 ##### save train 

def compute_recording_results():
    all_preds = []


    all_ys=[]

    tuh_splits = window_ds_set.split([[i] for i in range(len(window_train_set.datasets))])

    for rec_i, tuh_subset in tuh_splits.items():



        preds_rec=clf.predict_with_window_inds_and_ys(tuh_subset)

        all_preds.append(preds_rec['preds'].mean(axis=0))


        all_ys.extend([int(tuh_subset.description['pathological'][0])])


    mean_preds_per_trial_train = np.array(all_preds)#

    y =  window_train_set.description['pathological']
    assert (np.array(all_ys) == np.array(y)).all()

    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: mean_preds_per_trial_train[:, 0],
              column1: mean_preds_per_trial_train[:, 1],
              "true_pathological": y}

    assert len(y) == len(mean_preds_per_trial_train)


    pd.DataFrame.from_dict(a_dict).to_csv(result_path + "predictions_train_" + str(model_number) +
                                              ".csv")


    deep_preds_train =  mean_preds_per_trial_train[:, 0] <=  mean_preds_per_trial_train[:, 1]
    class_label_train = window_train_set.description['pathological']
    class_preds_train =deep_preds_train.astype(int)



    fig = plot_confusion_matrix_paper(confusion_matrix(class_label_train, class_preds_train),class_names=['normal', 'abnormal'])
    fig.savefig(result_path + 'confusion_matrix_train.png')



    confusion_mat_train = confusion_matrix(class_label_train, class_preds_train)
    precision_normal_train = round((confusion_mat_train.T[0][0] / float(np.sum(
        confusion_mat_train.T[0, :])))*100,2)


    precision_patho_train = round((confusion_mat_train.T[1][1] / float(np.sum(
        confusion_mat_train.T[1, :])))*100,2)


    recall_normal_train = round((confusion_mat_train.T[0][0] / float(np.sum(
                confusion_mat_train.T[:, 0])))*100,2)


    recall_patho_train = round((confusion_mat_train.T[1][1] / float(np.sum(
                confusion_mat_train.T[:, 1])))*100,2)


    accuracy_train= round((confusion_mat_train.T[0][0] + confusion_mat_train.T[1][1])/float(np.sum(confusion_mat_train.T[:, :])) *100, 2) # overall_accuracy

    precision_patho_train =precision_patho_train/100
    recall_normal_train = recall_normal_train/100
    precision_normal_train=precision_normal_train/100
    recall_patho_train=recall_patho_train/100

    f1_score_patho_train = round((2 * precision_patho_train * recall_patho_train / (precision_patho_train + recall_patho_train)),2)
    f1_score_patho_train

    f1_score_normal_train = round((2 * precision_normal_train * recall_normal_train / (precision_normal_train + recall_normal_train)),2)
    f1_score_normal_train

    results2 = {
        'model': model_name,
        'task_name': task_name,
        'abnormal train ' : len(window_train_set.description[window_train_set.description['pathological']==1]),
        'normal train ' : len(window_train_set.description[window_train_set.description['pathological']==0]),


        'precision_normal': precision_normal_train*100,
        'precision_patho': precision_patho_train*100,
        'recall_normal': recall_normal_train*100,
        'recall_patho': recall_patho_train*100,
        'f1_score_patho': f1_score_patho_train,
        'f1_score_normal': f1_score_normal_train,
        'accuracy': accuracy_train,
        'model_size':model_size,
        'cuda_memory_used': used_memory,#
        'time_used_in_s': foldtime,
    }

    all_results_train.append(results2)
    results_df = pd.DataFrame(all_results_train)
    fname = result_path +'_train_results.csv'
    results_df.to_csv(fname)
    print(f'Results saved under {fname}.')

    ## win acc

    win_preds =[]

    win_ys = []

    win_rec_number = []

    trial_preds=[]

    tuh_splits = window_train_set.split([[i] for i in range(len(window_train_set.datasets))])

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
    pd.DataFrame.from_dict(a_dict).to_csv(result_path +"prediction_windows_trainset.csv")


    deep_preds_train_win =  np.array(np.array(win_preds)[:, 0] <= np.array(win_preds)[:, 1])
    class_label_train_win  = np.array(win_ys)
    class_preds_train_win  =deep_preds_train_win.astype(int)


    fig = plot_confusion_matrix_paper(confusion_matrix(class_label_train_win, class_preds_train_win),class_names=['normal', 'abnormal'])
    fig.savefig(result_path + 'confusion_matrix_train_win.png')



    confusion_mat_train_win = confusion_matrix(class_label_train_win, class_preds_train_win)
    precision_normal_train_win = round((confusion_mat_train_win.T[0][0] / float(np.sum(
        confusion_mat_train_win.T[0, :])))*100,2)


    precision_patho_train_win = round((confusion_mat_train_win.T[1][1] / float(np.sum(
        confusion_mat_train_win.T[1, :])))*100,2)


    recall_normal_train_win = round((confusion_mat_train_win.T[0][0] / float(np.sum(
                confusion_mat_train_win.T[:, 0])))*100,2)


    recall_patho_train_win = round((confusion_mat_train_win.T[1][1] / float(np.sum(
                confusion_mat_train_win.T[:, 1])))*100,2)


    accuracy_train_win= round((confusion_mat_train_win.T[0][0] + confusion_mat_train_win.T[1][1])/float(np.sum(confusion_mat_train_win.T[:, :])) *100, 2) # overall_accuracy

    precision_patho_train_win =precision_patho_train_win/100
    recall_normal_train_win = recall_normal_train_win/100
    precision_normal_train_win=precision_normal_train_win/100
    recall_patho_train_win=recall_patho_train_win/100

    f1_score_patho_train_win = round((2 * precision_patho_train_win * recall_patho_train_win / (precision_patho_train_win + recall_patho_train_win)),2)
    f1_score_patho_train_win

    f1_score_normal_train_win = round((2 * precision_normal_train_win * recall_normal_train_win / (precision_normal_train_win + recall_normal_train_win)),2)
    f1_score_normal_train_win

    results2 = {
        'model': model_name,
        'task_name': task_name,
        'precision_normal': precision_normal_train_win*100,
        'precision_patho': precision_patho_train_win*100,
        'recall_normal': recall_normal_train_win*100,
        'recall_patho': recall_patho_train_win*100,
        'f1_score_patho': f1_score_patho_train_win,
        'f1_score_normal': f1_score_normal_train_win,
        'accuracy': accuracy_train_win,
        'model_size':model_size,
        'cuda_memory_used': used_memory,#=(torch.cuda.max_memory_allocated() / 1e9)
        'time_used_in_s': foldtime,
    }

    all_results_train_win.append(results2)
    results_df = pd.DataFrame(all_results_train_win)
    fname = result_path +'_train_win_results.csv'
    results_df.to_csv(fname)
    print(f'Results saved under {fname}.')


    ########### save valid set


    all_preds = []


    all_ys=[]

    tuh_splits = window_eval_set.split([[i] for i in range(len(window_eval_set.datasets))])

    for rec_i, tuh_subset in tuh_splits.items():



        preds_rec=clf.predict_with_window_inds_and_ys(tuh_subset)

        all_preds.append(preds_rec['preds'].mean(axis=0))


        all_ys.extend([int(tuh_subset.description['pathological'][0])])


    mean_preds_per_trial = np.array(all_preds)


    y =  window_eval_set.description['pathological']
    assert (np.array(all_ys) == np.array(y)).all()

    column0, column1 = "non-pathological", "pathological"
    a_dict = {column0: mean_preds_per_trial[:, 0],
              column1: mean_preds_per_trial[:, 1],
              "true_pathological": y}

    assert len(y) == len(mean_preds_per_trial)



    # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_path + "predictions_eval_" + str(model_number) +
                                              ".csv")


    deep_preds =  mean_preds_per_trial[:, 0] <=  mean_preds_per_trial[:, 1]
    class_label = window_eval_set.description['pathological']
    class_preds =deep_preds.astype(int)



    fig = plot_confusion_matrix_paper(confusion_matrix(class_label, class_preds),class_names=['normal', 'abnormal'])
    fig.savefig(result_path + 'confusion_matrix_eval.png')



    confusion_mat = confusion_matrix(class_label, class_preds)
    precision_normal = round((confusion_mat.T[0][0] / float(np.sum(
        confusion_mat.T[0, :])))*100,2)


    precision_patho = round((confusion_mat.T[1][1] / float(np.sum(
        confusion_mat.T[1, :])))*100,2)


    recall_normal = round((confusion_mat.T[0][0] / float(np.sum(
                confusion_mat.T[:, 0])))*100,2)


    recall_patho = round((confusion_mat.T[1][1] / float(np.sum(
                confusion_mat.T[:, 1])))*100,2)


    accuracy= round((confusion_mat.T[0][0] + confusion_mat.T[1][1])/float(np.sum(confusion_mat.T[:, :])) *100, 2) # overall_accuracy

    precision_patho =precision_patho/100
    recall_normal = recall_normal/100
    precision_normal=precision_normal/100
    recall_patho=recall_patho/100

    f1_score_patho = round((2 * precision_patho * recall_patho / (precision_patho + recall_patho)),2)
    f1_score_patho

    f1_score_normal = round((2 * precision_normal * recall_normal / (precision_normal + recall_normal)),2)
    f1_score_normal

    results2 = {
        'model': model_name,
        'task_name': task_name,
        'abnormal valid ' : len(window_eval_set.description[window_eval_set.description['pathological']==1]),
        'normal valid ' : len(window_eval_set.description[window_eval_set.description['pathological']==0]),
        'precision_normal': precision_normal*100,
        'precision_patho': precision_patho*100,
        'recall_normal': recall_normal*100,
        'recall_patho': recall_patho*100,
        'f1_score_patho': f1_score_patho,
        'f1_score_normal': f1_score_normal,
        'accuracy': accuracy,
        'model_size':model_size,
        'cuda_memory_used': used_memory,#
        'time_used_in_s': foldtime,
    }

    all_results_eval.append(results2)
    results_df = pd.DataFrame(all_results_eval)
    fname = result_path +'_eval_results.csv'
    results_df.to_csv(fname)
    print(f'Results saved under {fname}.')

    #### wind preds
    win_preds =[]

    win_ys = []

    win_rec_number = []

    trial_preds=[]

    tuh_splits = window_eval_set.split([[i] for i in range(len(window_eval_set.datasets))])

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
    pd.DataFrame.from_dict(a_dict).to_csv(result_path +"prediction_windows_evalset.csv")


    deep_preds_eval_win=  np.array(np.array(win_preds)[:, 0] <= np.array(win_preds)[:, 1])
    class_label_eval_win  = np.array(win_ys)
    class_preds_eval_win  =deep_preds_eval_win.astype(int)



    confusion_mat_eval_win = confusion_matrix(class_label_eval_win, class_preds_eval_win)
    precision_normal_eval_win = round((confusion_mat_eval_win.T[0][0] / float(np.sum(
        confusion_mat_eval_win.T[0, :])))*100,2)


    precision_patho_eval_win = round((confusion_mat_eval_win.T[1][1] / float(np.sum(
        confusion_mat_eval_win.T[1, :])))*100,2)


    recall_normal_eval_win = round((confusion_mat_eval_win.T[0][0] / float(np.sum(
                confusion_mat_eval_win.T[:, 0])))*100,2)


    recall_patho_eval_win = round((confusion_mat_eval_win.T[1][1] / float(np.sum(
                confusion_mat_eval_win.T[:, 1])))*100,2)


    accuracy_eval_win= round((confusion_mat_eval_win.T[0][0] + confusion_mat_eval_win.T[1][1])/float(np.sum(confusion_mat_eval_win.T[:, :])) *100, 2) # overall_accuracy

    precision_patho_eval_win =precision_patho_eval_win/100
    recall_normal_eval_win = recall_normal_eval_win/100
    precision_normal_eval_win=precision_normal_eval_win/100
    recall_patho_eval_win=recall_patho_eval_win/100

    f1_score_patho_eval_win = round((2 * precision_patho_eval_win * recall_patho_eval_win / (precision_patho_eval_win + recall_patho_eval_win)),2)
    f1_score_patho_eval_win

    f1_score_normal_eval_win = round((2 * precision_normal_eval_win * recall_normal_eval_win / (precision_normal_eval_win + recall_normal_eval_win)),2)
    f1_score_normal_eval_win

    results2 = {
        'model': model_name,

        'task_name': task_name,
        'precision_normal': precision_normal_eval_win*100,
        'precision_patho': precision_patho_eval_win*100,
        'recall_normal': recall_normal_eval_win*100,
        'recall_patho': recall_patho_eval_win*100,
        'f1_score_patho': f1_score_patho_eval_win,
        'f1_score_normal': f1_score_normal_eval_win,
        'accuracy': accuracy_eval_win,
        'model_size':model_size,
        'cuda_memory_used': used_memory,
        'time_used_in_s': foldtime,
    }

    all_results_eval_win.append(results2)
    results_df = pd.DataFrame(all_results_eval_win)
    fname = result_path +'_eval_win_results.csv'
    results_df.to_csv(fname)
    print(f'Results saved under {fname}.')



    del model, clf, window_train_set#, 



