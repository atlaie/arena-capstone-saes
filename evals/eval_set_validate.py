# %%
from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs
from inspect_ai.solver import multiple_choice

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RUN_NUMBER = 0

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
models = ['gemma-2-2b-it','gemma-2-2b-it-wmdp']

df = pd.DataFrame()

for model in models:
    log_dir = "logs-validation-" + model + str(RUN_NUMBER)
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
RUN_NUMBER += 1
#%%
df[['model_name','task','samples','accuracy','stderr']]

#%%

ax1 = df.plot(
    kind='bar',
    x="model_name",
    y="accuracy",
    )

ax1.set_ylim([0.45,0.6])

ax2 = df.plot(
    kind='bar',
    x="model_name",
    y="stderr",
    )

ax2.set_ylim([0,0.1])

plt.show()

#%%

## Same in Plotly
import plotly.express as px


fig = px.bar(df, x="model_name", y="accuracy",labels={"accuracy": "accuracy"})


fig.update_xaxes(range=[0.3, 0.7])  

fig.show()
#%%
# success, logs = eval_set(
#    tasks=["wmdp-fast.py"],
#    model=["hf/local"],
#    model_args={"model_path": "../gemma-2-2b-it-wmdp"},
#    log_dir="logs-validation1",
#    max_connections=32,
#    max_tokens=10
# )

# if success:
#     results = list_eval_logs("logs-validation1")
#     print(results)
# %%
