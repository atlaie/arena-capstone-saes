#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results2.csv')

# %%
df = df.replace({
    "gemma-2-2b-it-wmdp-random_ablated":"ablated_random",
    "gemma-2-2b-it-wmdp-target_ablated":"ablated_target",
    "gemma-2-2b-it":"base_instruct"
                 })
results = df[['task','model_name','accuracy','stderr']].sort_values(by='task')

# Define custom sorting for 'task' and 'model_name' columns
task_order = ['arc_easy', 'arc_challenge', 'gpqa_diamond', 'mmlu', 'wmdp']
model_order = ['base_instruct','ablated_random', 'ablated_target']

# Convert columns to categorical types with custom order
results['task'] = pd.Categorical(results['task'], categories=task_order, ordered=True)
results['model_name'] = pd.Categorical(results['model_name'], categories=model_order, ordered=True)

results['stderr'] = pd.to_numeric(results['stderr'], errors='coerce')


print(results)

custom_palette = {
    'base_instruct': '#A6C1E3',   # Low saturation blue
    'ablated_random': '#FFD4A3',  # Low saturation orange
    'ablated_target': '#FF8C42'   # More saturated orange
}


plt.figure(figsize=(12, 6))
ax = sns.barplot(data=results, 
            x='task', 
            y='accuracy', 
            hue='model_name',
            palette=custom_palette,
            # yerr=results['stderr'].values
            # yerr='stderr'
            )

## Data labels
# ax.bar_label(ax.containers[0])
# ax.bar_label(ax.containers[1])
# ax.bar_label(ax.containers[2])
    
# Customize labels and title
plt.xlabel('Task')
plt.ylabel('Accuracy')
plt.title('Accuracy by Task and Model Name')
plt.legend(title='Model Name')

# Show plot
plt.show()
# %%
