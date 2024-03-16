import os
import re
import pandas as pd
from typing import Dict, Optional, List
from utils.data_loader import XMLDataLoader
from utils.helpers import calc_metrics, plot_confusion_matrices
import argparse
from tqdm import tqdm
import numpy as np

PATH_TO_OFFICIAL_EVAL_SCRIPT: str = "./data/track1_eval_script/track1_eval.py"

def convert_df_preds_to_xmls(df_preds, path_to_output_dir: str):
    os.makedirs(path_to_output_dir, exist_ok=True)
    for patient_id in tqdm(df_preds['patient_id'].unique()):
        preds = df_preds[df_preds['patient_id'] == patient_id]
        tags: List[str] = '\n'.join([ f"<{criterion} met=\"{'met' if preds[preds['criterion'] == criterion].iloc[0]['is_met'] == 1 else 'not met'}\" />" for criterion in preds['criterion'].values ])
        with open(os.path.join(path_to_output_dir, f'{patient_id}.xml'), 'w') as f:
            f.write(f"""<?xml version="1.0" encoding="UTF-8" ?>
<PatientMatching>
<TEXT></TEXT>
<TAGS>
{tags}
</TAGS>
</PatientMatching>
""")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('path_to_csv', type=str, help='Path to eval.py output')
    args = parser.parse_args()
    return args

def gen_metrics(file_name: str, path_to_data: str):
    # Load data
    dataloader = XMLDataLoader(path_to_data)
    dataset = dataloader.load_data()
    id2patient: Dict[str, Dict] = { patient['patient_id']: patient for patient in dataset }

    # Load results.csv
    df_results = pd.read_csv(f"{file_name}.csv")
    
    # Force TRUE/FALSE to 1/0
    df_results['is_met']     = df_results['is_met'].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
    df_results['true_label'] = df_results['true_label'].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
    
    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")

    # Make preds by max pooling / taking most recent value for `is_met` across all notes
    df_preds = []
    for criterion in dataloader.criteria:
        if dataloader.criteria_2_agg[criterion] == 'max':
            # Max over each note's pred
            df_preds.append(df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'is_met' : 'max',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index())
        elif dataloader.criteria_2_agg[criterion] == 'min':
            # Max over each note's pred
            df_preds.append(df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'is_met' : 'min',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index())
        elif dataloader.criteria_2_agg[criterion].startswith('most_recent'):
            # Take most recent note's pred
            months: int = int(dataloader.criteria_2_agg[criterion].split("|")[-1])
            df_ = df_results[df_results['criterion'] == criterion].copy()
            df_['note_date'] = pd.to_datetime(df_.apply(lambda x : dataloader.get_date_of_note(id2patient[str(x['patient_id'])], x['note_idx']), axis=1))
            df_['current_date'] = pd.to_datetime(df_.apply(lambda x: dataloader.get_current_date_for_patient(id2patient[str(x['patient_id'])]), axis=1))
            df_['days_since'] = (df_['current_date'] - df_['note_date']).dt.days
            df_['is_within_time_period'] = df_['days_since'] <= months * 31
            df_ = df_[df_['is_within_time_period']]
            df_ = df_.groupby(['patient_id']).agg({
                'is_met' : 'max',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index()
            df_new = df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index()
            df_new['is_met'] = df_new.apply(lambda x: df_[df_['patient_id'] == x.patient_id]['is_met'].values[0] if x.patient_id in df_['patient_id'].unique() else 0, axis=1)
            df_new = df_new[['patient_id', 'true_label', 'is_met', 'criterion']]
            df_preds.append(df_new)
        else:
            raise ValueError(f"Unexpected aggregation method: {dataloader.criteria_2_agg[criterion]}")
    df_preds = pd.concat(df_preds)
    df_preds.to_csv(f"{file_name}_preds.csv")

    with open(f"{file_name}.log", 'w') as f:
        f1_scores: Dict[str, float] = {} # map from criterion to f1 score
        for criterion in dataloader.criteria:
            metrics: Dict[str, float] = calc_metrics(df_preds[df_preds['criterion'] == criterion]['true_label'].values,
                                                        df_preds[df_preds['criterion'] == criterion]['is_met'].values)
            f.write(f"{criterion}:\n")
            for k, v in metrics.items():
                f.write(f"\t{k}: {v}\n")
                if k == 'f1':
                    f1_scores[criterion] = v

        # Calculate overall metrics
        metrics: Dict[str, float] = calc_metrics(df_preds['true_label'].values,
                                                    df_preds['is_met'].values)
        f.write(f"Overall (Micro):\n")
        for k, v in metrics.items():
            f.write(f"\t{k}: {v}\n")
            if k == 'f1':
                f1_scores['overall-micro'] = v

    # Make confusion matrix for each criterion
    plot_confusion_matrices(df_preds, f"{file_name}_cm.png")
    
    # Save preds as XMLs so we can use the official scorer
    convert_df_preds_to_xmls(df_preds, f"{file_name}_xmls")
    
    # Run official scorer eval script
    os.system(f'python "{PATH_TO_OFFICIAL_EVAL_SCRIPT}" "{path_to_data}" "{file_name}_xmls" > "{file_name}_track1_eval.txt"')
    with open(f"{file_name}_track1_eval.txt", "r") as fd:
        contents: str = fd.read()
        micro_matches = re.search(r'Overall \(micro\).+    (\d+\.\d+)', contents)
        macro_matches = re.search(r'Overall \(macro\).+    (\d+\.\d+)', contents)
        if not micro_matches or not macro_matches:
            raise ValueError("ERROR - can't parse track1_eval.txt")
        micro: float = float(micro_matches.group(1))
        macro: float = float(macro_matches.group(1))

    # Make pretty table of F1 scores
    f1_scores['overall-micro-n2c2'] = micro
    f1_scores['overall-macro-n2c2'] = macro
    df_f1 = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['f1'])
    df_f1.to_csv(f"{file_name}_f1.csv")

    
def gen_metrics_koopman(file_name: str, path_to_data: str):
    # Load data
    dataloader = XMLDataLoader(path_to_data)
    dataset    = dataloader.load_data()
    id2patient: Dict[str, Dict] = { patient['patient_id']: patient for patient in dataset }

    # Load results.csv
    df_results = pd.read_csv(f"{file_name}.csv")
    
    # Force TRUE/FALSE to 1/0
    df_results['is_met']     = df_results['is_met'].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
    df_results['true_label'] = df_results['true_label'].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
    
    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")

    # Make preds by max pooling / taking most recent value for `is_met` across all notes
    df_preds = []
    for criterion in dataloader.criteria:
        if dataloader.criteria_2_agg[criterion] == 'max':
            # Max over each note's pred
            df_preds.append(df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'is_met' : 'max',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index())
        elif dataloader.criteria_2_agg[criterion] == 'min':
            # Max over each note's pred
            df_preds.append(df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'is_met' : 'min',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index())
        elif dataloader.criteria_2_agg[criterion].startswith('most_recent'):
            # Take most recent note's pred
            months: int = int(dataloader.criteria_2_agg[criterion].split("|")[-1])
            df_ = df_results[df_results['criterion'] == criterion].copy()
            df_['note_date'] = pd.to_datetime(df_.apply(lambda x : dataloader.get_date_of_note(id2patient[str(x['patient_id'])], x['note_idx']), axis=1))
            df_['current_date'] = pd.to_datetime(df_.apply(lambda x: dataloader.get_current_date_for_patient(id2patient[str(x['patient_id'])]), axis=1))
            df_['days_since'] = (df_['current_date'] - df_['note_date']).dt.days
            df_['is_within_time_period'] = df_['days_since'] <= months * 31
            df_ = df_[df_['is_within_time_period']]
            df_ = df_.groupby(['patient_id']).agg({
                'is_met' : 'max',
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index()
            df_new = df_results[df_results['criterion'] == criterion].groupby(['patient_id']).agg({
                'true_label' : 'first',
                'criterion' : 'first',
            }).reset_index()
            df_new['is_met'] = df_new.apply(lambda x: df_[df_['patient_id'] == x.patient_id]['is_met'].values[0] if x.patient_id in df_['patient_id'].unique() else 0, axis=1)
            df_new = df_new[['patient_id', 'true_label', 'is_met', 'criterion']]
            df_preds.append(df_new)
        else:
            raise ValueError(f"Unexpected aggregation method: {dataloader.criteria_2_agg[criterion]}")
    df_preds = pd.concat(df_preds)
    df_preds.to_csv(f"{file_name}_preds.csv")

    with open(f"{file_name}.log", 'w') as f:
        f1_scores: Dict[str, float] = {} # map from criterion to f1 score
        for criterion in dataloader.criteria:
            metrics: Dict[str, float] = calc_metrics(df_preds[df_preds['criterion'] == criterion]['true_label'].values,
                                                        df_preds[df_preds['criterion'] == criterion]['is_met'].values)
            f.write(f"{criterion}:\n")
            for k, v in metrics.items():
                f.write(f"\t{k}: {v}\n")
                if k == 'f1':
                    f1_scores[criterion] = v

        # Calculate overall metrics
        metrics: Dict[str, float] = calc_metrics(df_preds['true_label'].values,
                                                    df_preds['is_met'].values)
        f.write(f"Overall (Micro):\n")
        for k, v in metrics.items():
            f.write(f"\t{k}: {v}\n")
            if k == 'f1':
                f1_scores['overall-micro'] = v

    # Make confusion matrix for each criterion
    plot_confusion_matrices(df_preds, f"{file_name}_cm.png")
    
    # Save preds as XMLs so we can use the official scorer
    convert_df_preds_to_xmls(df_preds, f"{file_name}_xmls")
    
    # Run official scorer eval script
    os.system(f'python "{PATH_TO_OFFICIAL_EVAL_SCRIPT}" "{path_to_data}" "{file_name}_xmls" > "{file_name}_track1_eval.txt"')
    with open(f"{file_name}_track1_eval.txt", "r") as fd:
        contents: str = fd.read()
        micro_matches = re.search(r'Overall \(micro\).+    (\d+\.\d+)', contents)
        macro_matches = re.search(r'Overall \(macro\).+    (\d+\.\d+)', contents)
        if not micro_matches or not macro_matches:
            raise ValueError("ERROR - can't parse track1_eval.txt")
        micro: float = float(micro_matches.group(1))
        macro: float = float(macro_matches.group(1))

    # Make pretty table of F1 scores
    f1_scores['overall-micro-n2c2'] = micro
    f1_scores['overall-macro-n2c2'] = macro
    df_f1 = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['f1'])
    df_f1.to_csv(f"{file_name}_f1.csv")
if __name__ == '__main__':
    args = parse_args()
    file_name: str = args.path_to_csv.replace(".csv", "")
    split = 'train' if 'train' in file_name else 'test'
    path_to_data: str = './data/train/' if split == 'train' else './data/n2c2-t1_gold_standard_test_data/test/'

    gen_metrics(file_name, path_to_data)