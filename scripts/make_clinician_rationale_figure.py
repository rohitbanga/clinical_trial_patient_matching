import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.data_loader import XMLDataLoader
sns.set_theme(style="whitegrid")
sns.set_context("paper")

BASE_DIR: str = "./"
PATH_TO_OUTPUT_DIR: str = os.path.join(BASE_DIR, "figures/")

# Load data
path_to_results_xlsx: str = os.path.join(BASE_DIR, "outputs-results/rationale_results/CT Patient Matching - Rationales Dataset.xlsx")
dfs = pd.read_excel(path_to_results_xlsx, sheet_name=["Dev", "Jenelle"])
df = pd.concat(dfs, axis=0)
df.reset_index(drop=True, inplace=True)

# Load ground truth data
df_orig = pd.read_csv(os.path.join(BASE_DIR, "outputs-results/rationale_dataset_gpt4--each_criteria_all_notes/stratified_sample.csv"))

# Prepare the data
df = df.merge(df_orig, left_on=['Patient ID', 'Criterion'], right_on=['patient_id', 'criterion'], how='inner', suffixes=('', '_orig'))
df['is_correct'] = (df['is_met'] == df['true_label']).astype(int)
grouped = df.groupby(['Criterion', 'Assessment', 'is_correct']).size()
grouped.to_csv('./clinical_rationale.csv')

# Function to plot the bar chart
color_mapping = {
    'Correct': 'tab:green',
    'Partially Correct': 'tab:olive',
    'Incorrect': 'tab:red',
    np.nan : 'grey',
}
def plot_clustered_bar(ax, data, label):
    # Create sub-dataframe for the given label
    sub_data = data.xs(label, level='is_correct')

    # Unique criteria and assessments
    criteria = df['Criterion'].unique()
    assessments = df['Assessment'].unique()
    
    print(criteria)
    print(assessments)

    # Number of bars for each Criterion
    n_bars = len(assessments)

    # Positions of bars on x-axis
    barWidth = 0.2
    r = np.arange(len(criteria))
    positions = [r + barWidth*i for i in range(n_bars)]

    # Create bars
    for pos, assessment in zip(positions, assessments):
        counts = [sub_data.get((criterion, assessment), 0) for criterion in criteria]
        ax.bar(pos, counts, width=barWidth, edgecolor='grey', label=assessment, color=color_mapping[assessment])

    # General layout
    ax.set_xticks([r + barWidth*(n_bars-1)/2 for r in range(len(criteria))])
    ax.set_xticklabels(criteria, rotation=45)
    ax.set_xlabel('Criterion')
    ax.set_ylabel('Count')
    ax.set_title(f'{"Model was Correct" if label else "Model was Incorrect"}')
    ax.legend()

# Create a figure with two subplots
fig, ax = plt.subplots(figsize=(12, 6))
plot_clustered_bar(ax, grouped, 0)
plt.tight_layout()
plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, 'clinician_rationale_incorrect.png'))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
plot_clustered_bar(ax, grouped, 1)
plt.tight_layout()
plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, 'clinician_rationale_correct.png'))