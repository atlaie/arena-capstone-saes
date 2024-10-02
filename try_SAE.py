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

# Load the SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    #release="gemma-scope-2b-pt-res-canonical",
    release="gemma-scope-2b-pt-mlp",
    #sae_id="layer_25/width_16k/canonical",
    sae_id="layer_25/width_16k/average_l0_277",
    device=device
)

#%%
from torch.utils.data import DataLoader

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

def compute_errors(sae_in, feature_acts):
    # Flatten inputs and compute the mask for active features
    feature_acts = feature_acts.flatten(0,1)
    fired_mask = feature_acts.sum(dim=-1) > 0

    # Perform reconstruction
    reconstruction = feature_acts[fired_mask] @ sae.W_dec

    # Compute the reconstruction error (Sum of Squared Residuals, SSR)
    squared_residuals = (reconstruction - sae_in.flatten(0,1)[fired_mask]) ** 2
    ssr = squared_residuals.sum()

    # Compute the total variance of the original input (Total Sum of Squares, TSS)
    sae_in_flattened = sae_in.flatten(0,1)[fired_mask]
    mean_sae_in = sae_in_flattened.mean(dim=0)
    tss = ((sae_in_flattened - mean_sae_in) ** 2).sum()

    # Compute R^2
    r2 = 1 - ssr / tss

    # Return the mean squared reconstruction error and R^2
    return squared_residuals.mean(), r2


#%%

# Function to compute feature rank sums
def compute_feature_ranks_per_sample(model, sae, dataloader, device):
    model.eval()
    sae.eval()
    num_features = sae.cfg.d_sae
    all_ranks = []
    all_rec_errors = []
    all_r2 = []
    total_samples = 0

    for batch in tqdm(dataloader, desc="Processing"):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.size(0)
        total_samples += batch_size

        with torch.no_grad():
            # Run model to get activations at the SAE layer
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=sae.cfg.hook_name,
                prepend_bos=True
            )

            activations = cache[sae.cfg.hook_name]  # Shape: [batch_size, seq_len, hidden_size]
            
            # Encode activations using SAE
            feature_acts = sae.encode(activations)  # Shape: [batch_size, seq_len, num_features]
            
            reconstruction_error, r2 = compute_errors(activations, feature_acts)

            # Aggregate activations over positions (e.g., take max over seq_len)
            max_feature_acts = feature_acts.max(dim=1).values  # Shape: [batch_size, num_features]

            # Get sorted indices (features sorted by decreasing activation)
            sorted_indices = torch.argsort(-max_feature_acts, dim=-1)  # Shape: [batch_size, num_features]

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

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data = r2s, kde = True)
plt.xlabel(r"$R_{SAE \rightarrow MLP}^2$")
#%%
import seaborn as sns
import cmcrameri as cmc
idx_order = np.argsort(np.mean(feature_ranking.detach().cpu().numpy(), axis = 0))
f_rank = feature_ranking[:, idx_order].detach().cpu().numpy()
#%%
sns.heatmap(f_rank, cmap = 'cmc.lapaz', cbar_kws={'label':'Activation-based Rank'})
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
plt.xlim([-5, 105])
plt.ylim([-1, 20])
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
    
    # Sum the absolute values of the weights across the top latents
    # This gives us a measure of how much each neuron in the hidden layer is influenced
    neuron_importance = top_latent_weights.abs().sum(dim=0)  # Shape: [2304]
    
    perc = np.percentile(neuron_importance.detach().cpu().numpy(), percentile)

    return np.where(neuron_importance.detach().cpu().numpy() > perc)[0]


# Function to ablate the most important neurons in the model by zeroing out their downstream weights
def ablate_neurons_in_model(model, important_neurons):
    with torch.no_grad():
        # Get a copy of the down_proj weights (this will avoid in-place modifications)
        down_proj_weights = model.model.layers[-1].mlp.down_proj.weight.clone()
        
        # Zero out the rows corresponding to the important neurons
        down_proj_weights[important_neurons, :] = 0.0
        
        # Assign the modified weight back to the model
        model.model.layers[-1].mlp.down_proj.weight = torch.nn.Parameter(down_proj_weights)

    return model


#%%
weights_1 = model_transformers.model.layers[-1].mlp.down_proj.weight
important_neurons = identify_neurons_from_sae(sae, idx_order, top_k = 20, percentile=99)
important_neurons_random = np.random.choice(important_neurons.shape[0], size = important_neurons.shape[0], replace = False)
model_ablated = ablate_neurons_in_model(model_transformers, important_neurons)
model_ablated_random = ablate_neurons_in_model(model_transformers, important_neurons_random)
weights_ablated = model_ablated.model.layers[-1].mlp.down_proj.weight
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
plt.savefig('results/Ablated_Neurons_Weights_01102024.svg', transparent = True)
#%%
weights_ablated.shape
#%%
top_latent_weights = sae.W_dec[important_neurons, :]  # Shape: [20, 2304]
    
# Sum the absolute values of the weights across the top latents
# This gives us a measure of how much each neuron in the hidden layer is influenced
neuron_importance = top_latent_weights.abs().sum(dim=0)  # Shape: [2304]
sns.histplot(data = neuron_importance.detach().cpu().numpy(),
              bins = 50, kde = True, alpha = 0.7, color = 'grey', linewidth = 0.5)
perc = np.percentile(neuron_importance.detach().cpu().numpy(), 99)
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
