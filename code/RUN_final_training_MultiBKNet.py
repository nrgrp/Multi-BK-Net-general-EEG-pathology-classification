import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--trial", type=int)
parser.add_argument("--seed", type=int)

args = parser.parse_args()

trial = args.trial
seed = args.seed

import sys
sys.path.insert(0, "./code/")

from final_training_and_eval_MultiBKNet import * 



task_name='Final_train_and_eval_MultiBKNet'

save_path = f"./final_run/{task_name}/trial_{trial}/seed_{seed}/"

from pathlib import Path
Path(save_path).mkdir(parents=True, exist_ok=True)



df =  pd.read_csv(f"./{task_name}/config_trial_{trial}_params_{task_name}.csv")

objective_cv(trial, seed,save_path,task_name)
