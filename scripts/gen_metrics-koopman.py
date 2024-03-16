try: 
    import sys
    utils_path = "/pasteur/u/ale9806/Repositories/clinical_trial_patient_matching"
    sys.path.append(utils_path)
    print("added path")
except:
    pass


import os
import re
import pandas as pd
from typing import Dict, Optional, List
import json
from utils.metrics import calc_metrics, plot_confusion_matrices
import argparse
from tqdm import tqdm
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('path_to_csv', type=str, help='Path to eval.py output')
    args   = parser.parse_args()
    return args

def gen_metrics(file_name: str, path_to_data: str):
    # Load results.csv
    df_results = pd.read_csv(f"{file_name}.csv")

    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")
    df               = pd.read_csv(file_name)
    force_boolean(df,'is_met')
    force_boolean(df,'true_label')
    grouped_df       = df.groupby(['patient_id', 'trail_id'])

    preds           = []
    true_labels     = []
    
    for patient_id  in df["patient_id"].unique():
        sub_df = df[df["patient_id"] == patient_id]
        for trail_id in df["trail_id"].unique():
            sub_trail_df = sub_df[sub_df["trail_id"] == trail_id]
            inclussion_df:pd.DataFrame  = get_criterion_type(sub_df,'inclusion_criteria')
            exclussion_df:pd.DataFrame  = get_criterion_type(sub_df,'exclusion_criteria')
            meets:int                   = judge(inclussion_df,exclussion_df)

            preds.append(meets)
            true_labels.append(int(sub_trail_df["true_label"][0]))

    metrics  = calc_metrics(np.array(true_labels), np.array(preds))
    save_metrics_as_file(metrics,filename="out.txt")




        
        
def judge(inclussion_df,exclussion_df):
    all_inclussion:int = int(np.prod(inclussion_df["is_met"]))
    all_exclusion:int  = int(np.prod(exclussion_df["is_met"]))
    if all_inclussion == 1 and  all_exclusion == 0 :
        return 2
    else:
        return 0 
    
def force_boolean(df:pd.DataFrame,column:str) -> None:
     df[column] = df[column].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
        
def get_criterion_type(df:pd.DataFrame,criteria=str) -> pd.DataFrame:
    return df[df['criterion'].str.contains(criteria, case=False)]

def save_metrics_as_file(metrics_dict, filename):
    with open(filename+".log", 'w') as file:
        for key, value in metrics_dict.items():
            file.write(json.dumps({key: value}) + '\n')