import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--seed", type=int)
#parser.add_argument("--divisor",type=int)

args = parser.parse_args()

seed = args.seed

import sys
sys.path.insert(0, "/home/code/")


from train_and_eval_baseline_model import * 


model_name = 'Deep4Net'  #


input_window_samples=6000
stride_size=input_window_samples

drop_prob=0.5
batch_size=64
lr=0.01
n_epochs=35
weight_decay=0.0005



result_folder='/home/results/finalrun/'
train_folder = '/home/TUABCOMB/final_train/'
eval_folder ='/home/TUAB/final_eval/'
eval_folder2 ='/home/TUABEXB/final_eval/'
     
df = pd.read_csv('/home/TUABCOMB_trainset_5fold.csv')

 
ids_to_load_train = None 
ids_to_load_valid = None

task_name = f"TUABCOMB_finalrun_trialwise_winsize_{input_window_samples}_stride_{stride_size}"

train_and_eval_TUHEEG_pathology(model_name=model_name,
                     task_name = task_name,
                     drop_prob=drop_prob,
                     batch_size=batch_size,
                     lr=lr,
                     n_epochs=n_epochs,
                     weight_decay=weight_decay,
                     result_folder=result_folder,
                     train_folder=train_folder,
                     eval_folder=eval_folder,
                     eval_folder2=eval_folder2,
                     ids_to_load_train =  ids_to_load_train,
                     stride_size=stride_size,
                     input_window_samples=input_window_samples,
                     cuda = True,
                     seed= seed,
                     )