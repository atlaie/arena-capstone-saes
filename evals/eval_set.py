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

fast = True
fast_string = "fast-" if fast else ""
# only used if fast is True
use_ratio = True
ratio = 0.01
max_sample = 'inf'
min_sample = 100
with open('eval_config.json', 'w') as f:
    json.dump({'use_ratio':use_ratio, 'ratio':ratio, 'max_sample':max_sample, 'min_sample':min_sample}, f)

while os.path.isdir("logs-validation-full-" + fast_string + str(RUN_NUMBER)) and RUN_NUMBER < 100:
    RUN_NUMBER += 1

#%%

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
# run evals
models = ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-random_ablated']
models += ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-target_ablated']
models += ['gemma-2-2b-it']

df = pd.DataFrame()
tasks = ["wmdp_fast.py","arc_fast.py","gpqa_fast.py","mmlu_fast.py","mmlu-pro_fast.py"] if fast else ["wmdp.py","arc.py","gpqa.py","mmlu.py","mmlu-pro.py"]


for model in models:
    log_dir = "logs-validation-full-" + fast_string + str(RUN_NUMBER) + "/" + model.split('/')[-1]
    success, logs = eval_set(
        tasks=tasks,
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

# # log_dir = "logs-validation2"

# models = ['gemma-2-2b-it']
# models += ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-random_ablated']
# models += ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-target_ablated']

# df = pd.DataFrame()
# fast = False
# tasks = ["wmdp_fast.py","arc_fast.py","gpqa_fast.py","mmlu_fast.py","mmlu-pro_fast.py"] if fast else ["wmdp.py","arc.py","gpqa.py","mmlu.py","mmlu-pro.py"]

# for model in models:
#     log_dir = "logs-validation-" + str(RUN_NUMBER) + "/" + model.split('/')[-1]
#     success, logs = eval_set(
#         tasks=tasks,
#         model=["hf/local"],
#         model_args={ "model_path": f"../{model}"},
#         log_dir=log_dir,
#         max_connections=32,
#         max_tokens=10
#         )
    
#     results = process_json_files(log_dir)
#     df = pd.concat([df, results])
#     df.dropna(inplace=True)

# df['model_name'] = df['model_path'].apply(lambda x: x.split('/')[-1] if pd.notna(x) else 'Unknown')
# df['samples'] = df['samples'].astype(int)
# print(df[['model_name','task','samples','accuracy','stderr']])
# RUN_NUMBER += 1

#%%

# df[['task','model_name','samples','accuracy','stderr']].sort_values(by='task')
#%%

# ax1 = df.plot(
#     kind='bar',
#     x="model_name",
#     y="accuracy",
#     )

# # ax1.set_ylim([0.45,0.6])

# # ax2 = df.plot(
# #     kind='bar',
# #     x="model_name",
# #     y="stderr",
# #     )

# # ax2.set_ylim([0,0.1])

# plt.show()
#%%

# ORIGINAL
# models = ['gemma-2-2b-it-wmdp/gemma-2-2b-it']

# success, logs = eval_set(
# #    tasks=["gpqa.py"],
#    tasks=["wmdp.py","arc.py","gpqa.py","mmlu.py","mmlu-pro.py"],
#    model=["hf/local"],
#    model_args={"model_path": "../gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-ablated"},
#    log_dir="logs-run-fine-tuned-clean",
#    max_connections=32,
#    max_tokens=10,
#    max_samples=100
# )

# if success:
#     results = list_eval_logs("logs-run-test-2")
#     print(results)
# %%

# ABLATED
# models = ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-ablated']

# success, logs = eval_set(
# #    tasks=["gpqa.py"],
#    tasks=["wmdp_fast.py","arc_fast.py","gpqa_fast.py","mmlu_fast.py","mmlu-pro_fast.py"],
#    model=["hf/local"],
#    model_args={"model_path": "../gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-ablated"},
#    log_dir="logs-run-fine-tuned-clean-sad",
#    max_connections=32,
#    max_tokens=10,
#    max_samples=100
# )

# if success:
#     results = list_eval_logs("logs-run-test-2")
#     print(results)
# %%
