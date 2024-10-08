#%%
from sae_lens import SAE
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import numpy as np
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = HookedTransformer.from_pretrained("google/gemma-2-2b-it", device=device)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')


from torch.utils.data import DataLoader
import random
# Load and process the sWMDP Dataset
full_synthetic_wmdp = pd.read_csv("full_synthetic_wmdp.csv")

DATA_SEED = 42
train_data = full_synthetic_wmdp.sample(frac=1, random_state=DATA_SEED)
train_data.reset_index(drop=True, inplace=True)

# Convert the Pandas DataFrame to Hugging Face Dataset format
dataset_train = Dataset.from_pandas(train_data)

def tokenize_function_with_choices(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for i in range(len(examples["choices"])):
        # Clean the choices string for the current example
        raw_choices = examples["choices"][i].strip("[]")  # Remove the square brackets
        choices_list = [choice.strip("' ") for choice in raw_choices.split(", ")]  # Clean and split

        # Combine the question and the cleaned-up choices into a single input string
        combined_input = (
            examples["question"][i] + "\nChoices: " + ', '.join(choices_list) + "\nAnswer:"
        )

        # Tokenize the combined input
        tokenized_inputs = tokenizer(
            combined_input,
            truncation=True,
            max_length=512
        )

        # Get the correct answer based on the index
        answer_idx = examples["answer"][i]  # The index of the correct answer
        correct_answer = choices_list[answer_idx]  # Retrieve the correct answer text

        # Tokenize the correct answer (as the label for the model to predict)
        labels = tokenizer(
            correct_answer,
            truncation=True,
            max_length=512
        )["input_ids"]

        # Mask out padding tokens (-100 to ignore them in loss calculation)
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]

        # Append the tokenized data to the respective lists
        input_ids_list.append(tokenized_inputs["input_ids"])
        attention_mask_list.append(tokenized_inputs["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# Tokenize the training set and remove unnecessary columns
tokenized_dataset_train = dataset_train.map(
    tokenize_function_with_choices,
    batched=True,
    remove_columns=dataset_train.column_names  # Remove original columns
)

# Initialize the Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    return_tensors="pt",
    label_pad_token_id=-100
)

# Define the subset size
subset_size = 1024  # Adjust this number as needed

# Select a subset of the dataset
subset_dataset = tokenized_dataset_train.select(range(subset_size))

# Create DataLoader with the subset
batch_size = 32  # Adjust based on your GPU memory
train_dataloader = DataLoader(
    subset_dataset,
    #tokenized_dataset_train,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

import einops
import torch as t

def compute_logit_attribution(model, layer_activations, correct_toks, incorrect_toks):
    # Get logits in the "Correct - Incorrect" direction, of shape (4, d_model)
    logit_direction = model.W_U.T[correct_toks] - model.W_U.T[incorrect_toks]
    
    # Get the last (in seq) layer activations from the model
    layer_acts_post = layer_activations[:, -1]  # Shape: [batch_size, d_model]

    # Get DLA by computing the dot product of the residual directions onto the logit direction
    # Shape: [batch_size, d_sae]
    dla = einops.einsum(layer_acts_post, logit_direction, 'b d_model, b d_model -> d_model')

    return dla

all_layers = []
for elem in list(model.named_modules()):
    all_layers.append(elem[0])


def filter_out_substrings(input_list, substring):
    return [s for s in input_list if substring in s]

model.eval()
all_ranks = []
all_rec_errors = []
all_r2 = []
total_samples = 0
counter = 0
mlp_names = filter_out_substrings(all_layers, 'hook_mlp_out')
all_logits = t.zeros((int(len(subset_dataset)/batch_size),len(mlp_names), model.cfg.d_model))
for batch in tqdm(train_dataloader, desc="Processing"):
    # Move data to device
    input_ids = batch["input_ids"].to(device)
    batch_size = input_ids.size(0)
    total_samples += batch_size

    correct_toks = []
    incorrect_toks = []

    with torch.no_grad():
        for i in range(batch_size):
            # Retrieve the choices from the original examples (already tokenized)
            choices = dataset_train['choices'][i]
            answer_idx = dataset_train['answer'][i]

            # Get the correct answer token (already tokenized in `labels`)
            correct_answer = choices[answer_idx]
            correct_token_ids = tokenizer.encode(correct_answer, truncation=True, max_length=512)

            # Get a random incorrect answer token (make sure it's not the correct one)
            incorrect_choices = [choice for j, choice in enumerate(choices) if j != answer_idx]
            random_incorrect_answer = random.choice(incorrect_choices)
            incorrect_token_ids = tokenizer.encode(random_incorrect_answer, truncation=True, max_length=512)

            # Store correct and incorrect tokens for each sample
            correct_toks.append(correct_token_ids[1])  
            incorrect_toks.append(incorrect_token_ids[1])  

        # Run model to get activations at the SAE layer
        _, cache = model.run_with_cache(
            input_ids,
            prepend_bos=True
        )
        
        logits = t.zeros((len(mlp_names), model.cfg.d_model))
        for ii, name in enumerate(mlp_names):
            activations = cache[name]  # Shape: [batch_size, seq_len, hidden_size]
        
            all_logits[counter, ii,:] = compute_logit_attribution(model, activations, correct_toks, incorrect_toks)
    counter += 1

#%%
import seaborn as sns
import cmcrameri as cmc
import matplotlib.pyplot as plt
sorted_logits = np.mean(np.sort(all_logits.detach().cpu().numpy(), axis = -1)[...,-50:], axis = 0)
plt.subplots(figsize = (6,5))
palette = sns.color_palette('cmc.managua', n_colors=sorted_logits.shape[0])  # Seaborn palette
# Create a list of x values (Layer #) and corresponding medians
layer_indices = np.arange(sorted_logits.shape[0])  # X values are the layer indices
median_values = np.mean(sorted_logits, axis=1)   # Y values are the means per layer

# Plot each point with a specific color from the palette (including individual logits)
for i, logit in enumerate(sorted_logits):
    plt.semilogy(i + 0.1 * np.random.random(size=sorted_logits.shape[1]), logit, 'o', 
                 mec=palette[i], mfc='white', alpha=0.7)

# Plot the means with black markers
for i, mean_val in enumerate(mean_values):
    if i == np.argmax(mean_values):
        mss = 12
        mfcs = sns.desaturate('orange', 0.6)
    else:
        mss = 7
        mfcs = 'grey'
    plt.semilogy(i, mean_val, 'o', mec='black', mfc=mfcs, ms=mss)

# Set y-axis to log scale
plt.yscale('log')
# Fit and plot the linear regression to the means only
sns.regplot(x=layer_indices, y=mean_values, scatter=False, ci=90, color='grey', truncate=False)
plt.xlabel('Layer #', fontsize = 14)
plt.ylabel('Direct Logit Attribution', fontsize = 14)
plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
plt.ylim(1, 3e2)
sns.despine()
plt.tight_layout()
#plt.savefig('LogitAttribution_MLPlayers_03102024.svg', transparent = True)
#%%


cache[filter_out_substrings(cache.keys(), 'mlp.hook_post')[0]]
#%%
import random
# Function to compute feature rank sums
def compute_feature_ranks_per_sample(model, sae, dataloader, device):

    model.eval()
    all_ranks = []
    all_rec_errors = []
    all_r2 = []
    total_samples = 0

    for batch in tqdm(dataloader, desc="Processing"):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.size(0)
        total_samples += batch_size

        correct_toks = []
        incorrect_toks = []

        with torch.no_grad():
            for i in range(batch_size):
                # Retrieve the choices from the original examples (already tokenized)
                choices = dataset_train['choices'][i]
                answer_idx = dataset_train['answer'][i]

                # Get the correct answer token (already tokenized in `labels`)
                correct_answer = choices[answer_idx]
                correct_token_ids = tokenizer.encode(correct_answer, truncation=True, max_length=512)

                # Get a random incorrect answer token (make sure it's not the correct one)
                incorrect_choices = [choice for j, choice in enumerate(choices) if j != answer_idx]
                random_incorrect_answer = random.choice(incorrect_choices)
                incorrect_token_ids = tokenizer.encode(random_incorrect_answer, truncation=True, max_length=512)

                # Store correct and incorrect tokens for each sample
                correct_toks.append(correct_token_ids[1])  
                incorrect_toks.append(incorrect_token_ids[1])  

            # Run model to get activations at the SAE layer
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=sae.cfg.hook_name,
                prepend_bos=True
            )
            logits = []
            for ii, name in enumerate(filter_out_substrings(cache.keys(), 'mlp.hook_post')):
                activations = cache[sae.cfg.hook_name]  # Shape: [batch_size, seq_len, hidden_size]
            
                logit_attributions = compute_logit_attribution(model, activations, correct_toks, incorrect_toks)
                logits.append(logit_attributions)

            # Aggregate activations over positions (e.g., take max over seq_len)
            #max_feature_acts = feature_acts.max(dim=1).values  # Shape: [batch_size, num_features]

            # Get sorted indices (features sorted by decreasing activation)
            sorted_indices = torch.argsort(-logit_attributions, dim=-1)  # Shape: [batch_size, num_features]

            # Initialize ranks tensor with integer type
            ranks = torch.zeros_like(sorted_indices)  # dtype will be torch.int64

            # Create rank values (dtype will be torch.int64)
            rank_values = torch.arange(num_features, device=device).unsqueeze(0).expand(batch_size, -1)

            # Scatter rank values into ranks tensor
            ranks.scatter_(dim=1, index=sorted_indices, src=rank_values)

            # Move ranks to CPU and append to list
            ranks_cpu = ranks.cpu()
            all_ranks.append(ranks_cpu)
            all_rec_errors.append(reconstruction_error)
            all_r2.append(r2)

            # Clean up to free memory
            del cache, feature_acts, activations, reconstruction_error, r2
        
        #Free some more memory
        input_ids.to('cpu')

    # Concatenate all ranks along 0th dimension
    ranks_matrix = torch.cat(all_ranks, dim=0)  # Shape: [n_samples, num_features]
    rec_errors = torch.tensor(all_rec_errors)  # Shape: [n_samples]
    r2s = torch.tensor(all_r2)  # Shape: [n_samples]

    return ranks_matrix, rec_errors, r2s

# Run computation
feature_ranking, rec_errors, r2s = compute_feature_ranks_per_sample(
    model=model,
    sae=sae,
    dataloader=train_dataloader,
    device=device
)
idx_order = np.argsort(np.mean(feature_ranking.detach().cpu().numpy(), axis = 0))

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data = r2s, kde = True)
plt.xlabel(r"$R_{SAE \rightarrow MLP}^2$")
#%%
import seaborn as sns
import cmcrameri as cmc
f_rank = feature_ranking[:, idx_order].detach().cpu().numpy()
#%%
sns.heatmap(f_rank, cmap = 'cmc.lapaz', cbar_kws={'label':'DLA-based Rank'})
plt.xlabel('SAE latent #')
plt.ylabel('Dataset sample #')
plt.tight_layout()
#plt.savefig('results/LatentOrder_01102024.png', transparent = True, dpi = 200)
#%%
import matplotlib.pyplot as plt

plt.plot(np.std(f_rank, axis = 0)/np.sqrt(f_rank.shape[0]), color = 'gray', alpha = 1, lw = 1)
plt.xlabel('SAE latent #')
plt.ylabel('$\sigma(A)/\sqrt{N_{SAE}}$')
plt.axvline(20, ls = '--', lw = 1, color = 'black')
sns.despine()
plt.tight_layout()
#plt.xlim([-5, 105])
#plt.ylim([-1, 20])
#plt.savefig('results/LatentStd_ZoomedIn_01102024.svg', transparent = True)

#%%
from tabulate import tabulate

for ii in range(3):
    logits = sae.W_dec[idx_order[ii]] @ model.W_U

    top_logits, top_token_ids = logits.topk(5)
    top_tokens = model.to_str_tokens(top_token_ids)
    bottom_logits, bottom_token_ids = logits.topk(5, largest=False)
    bottom_tokens = model.to_str_tokens(bottom_token_ids)

    print(
        tabulate(
            zip(map(repr, bottom_tokens), bottom_logits, map(repr, top_tokens), top_logits),
            headers=["Bottom Tokens", "Logits", "Top Tokens", "Logits"],
            tablefmt="simple_outline",
            stralign="right",
            floatfmt="+.4f",
            showindex=True,
        )
    )

#%%
from transformers import AutoModelForCausalLM

model_transformers = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it')

#%%

# Function to identify which neurons in the hidden layer are most affected by the top latents
def identify_neurons_from_sae(sae, idx_order, top_k = 20, percentile = 99):
    # Select the rows of W_dec corresponding to the top latents
    top_latent_weights = sae.W_dec[idx_order[:top_k], :]  # Shape: [20, 2304]
    
    # L0 measure of the weights across the top latents, as a measure of how much each 
    # neuron in the hidden layer is influenced by these SAE latents.
    neuron_importance = top_latent_weights.abs().sum(dim=0)  # Shape: [2304]
    
    perc = np.percentile(neuron_importance.detach().cpu().numpy(), percentile)

    return np.where(neuron_importance.detach().cpu().numpy() > perc)[0]


# Function to ablate the most important neurons in the model by zeroing out their downstream weights
def ablate_neurons_in_model(model, important_neurons, layer):
    with torch.no_grad():
        # Get a copy of the down_proj weights (this will avoid in-place modifications)
        down_proj_weights = model.model.layers[layer].mlp.down_proj.weight.clone()
        
        # Zero out the rows corresponding to the important neurons
        down_proj_weights[important_neurons, :] = 0.0
        
        # Assign the modified weight back to the model
        model.model.layers[layer].mlp.down_proj.weight = torch.nn.Parameter(down_proj_weights)

    return model


#%%
#weights_1 = model_transformers.model.layers[-1].mlp.down_proj.weight
important_neurons = identify_neurons_from_sae(sae, idx_order, top_k = 50, percentile=95)
important_neurons_random = np.random.choice(important_neurons.shape[0], size = important_neurons.shape[0], replace = False)
model_ablated = ablate_neurons_in_model(model_transformers, important_neurons, layer = 12)
model_ablated_random = ablate_neurons_in_model(model_transformers, important_neurons_random, layer = 12)
#weights_ablated = model_ablated.model.layers[-1].mlp.down_proj.weight
#%%
important_neurons.shape
#%%
import matplotlib.pyplot as plt
plt.subplots(figsize = (15, 5))
sums = np.sum(weights_ablated.detach().numpy(), axis = 1)
plt.plot(sums, lw = 1, zorder = -30, color = 'grey')
zero_w = np.where(sums == 0)[0]
for ii, line in enumerate(zero_w):
    plt.axvline(line, color = 'black', lw = 1, ls = '--')
    if ii == 0:
        plt.plot(line, sums[line], 'o', color = 'black', mfc = 'black', zorder = -1, ms = 6, label = 'Ablated')
    plt.plot(line, sums[line], 'o', color = 'black', mfc = 'black', zorder = -1, ms = 6)

plt.legend(handlelength = 0, labelcolor = 'linecolor', frameon = False, fontsize = 14)
plt.ylabel('$W_{MLP}$', fontsize = 14)
plt.xlabel('MLP neuron #', fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
sns.despine()
plt.tight_layout()
#plt.savefig('results/Ablated_Neurons_Weights_01102024.svg', transparent = True)
#%%
weights_ablated.shape
#%%
import seaborn as sns
import matplotlib.pyplot as plt
top_latent_weights = sae.W_dec[important_neurons, :]  # Shape: [20, 2304]
    
# Sum the absolute values of the weights across the top latents
# This gives us a measure of how much each neuron in the hidden layer is influenced
neuron_importance = top_latent_weights.abs().sum(dim=0)  # Shape: [2304]
sns.histplot(data = neuron_importance.detach().cpu().numpy(),
              bins = 50, kde = True, alpha = 0.7, color = 'grey', linewidth = 0.5)
perc = np.percentile(neuron_importance.detach().cpu().numpy(), 95)
plt.axvline(perc, color = 'black')
plt.xlabel(r"Importance $(||W_{SAE\rightarrow Neuron}||_0)$")
sns.despine()
plt.tight_layout()
#plt.savefig('results/NeuronSAE_Importance_01102024.svg', transparent = True)

#%%

#%%
model_ablated.save_pretrained("gemma-2-2b-it-wmdp-target_ablated")
model_ablated_random.save_pretrained("gemma-2-2b-it-wmdp-random_ablated")

# %%
