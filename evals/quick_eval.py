#%%
from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs
from inspect_ai.solver import multiple_choice

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    eval_data = data.get('eval', {})
    results_data = data.get('results', {})
    scores_data = results_data.get('scores', [{}])[0].get('metrics', {})
    
    return {
        'task': eval_data.get('task'),
        'samples': eval_data.get('dataset', {}).get('samples'),
        'dataset_name': eval_data.get('dataset', {}).get('name'),
        'model_path': eval_data.get('model_args', {}).get('model_path'),
        'accuracy': scores_data.get('accuracy', {}).get('value'),
        'stderr': scores_data.get('stderr', {}).get('value')
    }

def process_json_files(directory):
    data_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            data = extract_data_from_json(file_path)
            data_list.append(data)
    
    return pd.DataFrame(data_list)

#%%

# log_dir = "logs-validation2"
models = ['gemma-2-2b-it-wmdp','gemma-2-2b-it_noLoRa']

df = pd.DataFrame()

for model in models:
    log_dir = "logs-validation-" + model
    success, logs = eval_set(
        tasks=["wmdp-fast.py"],
        model=["hf/local"],
        model_args={ "model_path": f"../{model}"},
        log_dir=log_dir,
        max_connections=32,
        max_tokens=10
        )
    
    results = process_json_files(log_dir)
    df = pd.concat([df, results])
    df.dropna(inplace=True)

df['model_name'] = df['model_path'].apply(lambda x: x.split('/')[-1] if pd.notna(x) else 'Unknown')
df['samples'] = df['samples'].astype(int)
print(df[['model_name','task','samples','accuracy','stderr']])