# %%
from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs
from inspect_ai.solver import multiple_choice

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


RUN_NUMBER = 0
# task datasets aren't guaranteed to shuffle if fast is false
fast = True
fast_string = "fast-" if fast else ""
# only used if fast is True
use_ratio = True
ratio = 0.01
max_sample = 'inf' # max sample puts  acap on the ratio if it is used
min_sample = 512 # min_sample takes priority over everything else
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
# models = ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-allMLPs_ablated']
models = ['gemma-2-2b-it-wmdp-target_ablated_SUSS']
# models = ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-target_ablated']
# models += ['gemma-2-2b-it-wmdp/gemma-2-2b-it-wmdp-random_ablated']
# models += ['gemma-2-2b-it'] # don't need baseline?

df = pd.DataFrame()
tasks = ["wmdp_fast.py"]
# tasks = ["wmdp_fast.py","arc_fast.py","gpqa_fast.py","mmlu_fast.py"] if fast else ["wmdp.py","arc.py","gpqa.py","mmlu.py"]


# results = process_json_files('logs-validation-full-0/gemma-2-2b-it')
# df = pd.concat([df, results])
# df.dropna(inplace=True)

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

df_sorted = df[['task','model_name','samples','accuracy','stderr']].sort_values(by='task')

df = df.replace({
    "gemma-2-2b-it-wmdp-random_ablated":"ablated_random",
    "gemma-2-2b-it-wmdp-target_ablated":"ablated_target",
    # "gemma-2-2b-it-wmdp-layer_ablated":"ablated_target",
    "gemma-2-2b-it":"base_instruct",
                 })
results = df[['task','model_name','accuracy','stderr']].sort_values(by='task')

# Define custom sorting for 'task' and 'model_name' columns
task_order = ['wmdp_bio', 'wmdp_cyber', 'wmdp_chem']
# task_order = ['arc_easy', 'arc_challenge', 'gpqa_diamond', 'mmlu', 'wmdp_bio', 'wmdp_cyber', 'wmdp_chem']
model_order = ['base_instruct','ablated_random', 'ablated_target']

# Convert columns to categorical types with custom order
results['task'] = pd.Categorical(results['task'], categories=task_order, ordered=True)
results['model_name'] = pd.Categorical(results['model_name'], categories=model_order, ordered=True)

results['stderr'] = pd.to_numeric(results['stderr'], errors='coerce')

custom_palette = {
    'base_instruct': '#A6C1E3',   # Low saturation blue
    'ablated_random': '#FFD4A3',  # Low saturation orange
    'ablated_target': '#FF8C42'   # More saturated orange
}
#%%

# plt.figure(figsize=(12, 6))
# ax = sns.barplot(data=results, 
#             x='task', 
#             y='accuracy', 
#             hue='model_name',
#             palette=custom_palette,
#             # yerr=results['stderr'].values
#             # yerr='stderr'
#             )
#%%
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=results, 
            x='task', 
            y='accuracy', 
            hue='model_name',
            palette=custom_palette,
            # yerr=results['stderr'].values
            # yerr='stderr'
            )
ax.errorbar(x = results['task'], y = results['accuracy'], yerr = results['stderr'], ls = '-')
    
# Customize labels and title
plt.xlabel('Task')
plt.ylabel('Accuracy')
plt.title('Accuracy by Task and Model Name')
plt.legend(title='Model Name')

# Show plot
#plt.savefig("logs-validation-full-" + fast_string + str(RUN_NUMBER) + "/" + "results.png")
plt.show()

RUN_NUMBER += 1
# %%


# ax1 = df_sorted.plot(
#     kind='bar',
#     x="model_name",
#     y="accuracy",
#     )

# # # ax1.set_ylim([0.45,0.6])

# # # ax2 = df.plot(
# # #     kind='bar',
# # #     x="model_name",
# # #     y="stderr",
# # #     )

# # # ax2.set_ylim([0,0.1])

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