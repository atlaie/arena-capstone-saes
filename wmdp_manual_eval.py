# %%
# Import necessary libraries
from sae_lens import SAE
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import numpy as np
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import einops
import random
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = HookedTransformer.from_pretrained("google/gemma-2-2b-it", device=device)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')

# Load the SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-mlp",
    sae_id="layer_24/width_16k/average_l0_357",
    device=device
)

# Load and process the sWMDP Dataset
full_synthetic_wmdp = pd.read_csv("full_synthetic_wmdp.csv")
DATA_SEED = 42
train_data = full_synthetic_wmdp.sample(frac=1, random_state=DATA_SEED)
train_data.reset_index(drop=True, inplace=True)
dataset_train = Dataset.from_pandas(train_data)

GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)

# Tokenization
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

def tokenize_function_with_choices(examples):
    input_ids_list = []
    attention_mask_list = []
    correct_answers = []

    for i in range(len(examples["choices"])):
        raw_choices = examples["choices"][i].strip("[]")
        choices_list = [choice.strip("' ") for choice in raw_choices.split(", ")]
        question = examples["question"][i]
        
        choice_letters = [' '+chr(65 + j) for j in range(len(choices_list))]
        correct_answer = tokenizer(choice_letters[dataset_train["answer"][1]], truncation=True, max_length=2, padding="max_length")
        choices_formatted = "\n".join([f"{letter}: {choice}" for letter, choice in zip(choice_letters, choices_list)])

        combined_input_template = SINGLE_ANSWER_TEMPLATE.format(
            letters=", ".join(choice_letters),
            question=question,
            choices=choices_formatted
        )

        tokenized_inputs = tokenizer(combined_input_template, truncation=True, max_length=512, padding="max_length")
        input_ids_list.append(tokenized_inputs["input_ids"])
        attention_mask_list.append(tokenized_inputs["attention_mask"])
        correct_answers.append(correct_answer)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": correct_answers
    }

# Tokenize the training set and remove unnecessary columns
tokenized_dataset_train = dataset_train.map(
    tokenize_function_with_choices,
    batched=True,
    remove_columns=dataset_train.column_names
)

# DataLoader Custom Collate Function
def custom_collate_fn(batch):
    # Stack input_ids and attention_mask
    input_ids = torch.stack([torch.tensor(ex['input_ids']) for ex in batch])
    attention_mask = torch.stack([torch.tensor(ex['attention_mask']) for ex in batch])

    # Stack the tokenized correct answers (labels)
    # Since labels are tokenized and might be sequences, we need to stack them as tensors
    labels = torch.stack([torch.tensor(ex['labels']['input_ids']) for ex in batch])

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Define the subset size
subset_size = 16#len(tokenized_dataset_train)  # Adjust this number as needed

# Select a subset of the dataset
subset_dataset = tokenized_dataset_train.select(range(subset_size))

# Create DataLoader
batch_size = 8
train_dataloader = DataLoader(
    subset_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn
)

#%%

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

import einops
import random

def compute_logit_attribution(model, sae_acts, correct_toks, incorrect_toks):
    # Get logits in the "Correct - Incorrect" direction, of shape (4, d_model)
    logit_direction = model.W_U.T[correct_toks] - model.W_U.T[incorrect_toks]
    
    # Get last (in seq) latent activations
    sae_acts_post = sae_acts[:, -1]

    # Calculate the residual contribution from the MLP (sae -> MLP activations -> residual)
    sae_resid_dirs = einops.einsum(
                                    sae_acts_post,
                                    sae.W_dec,   # Decoder for the latent space back to the hidden dimensions
                                    "batch d_sae, d_sae d_model -> batch d_sae d_model"
                                    )

    # Get DLA by computing average dot product of each latent's residual dir onto the logit dir
    dla = (sae_resid_dirs * logit_direction[:, None, :]).sum(-1)

    return dla


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

        correct_toks = []
        incorrect_toks = []

        with torch.no_grad():
            for i in range(batch_size):

                # Retrieve the tokenized choices and format them as in the tokenization function
                raw_choices = dataset_train['choices'][i].strip("[]")  # Clean the choices string for the current example
                choices_list = [choice.strip("' ") for choice in raw_choices.split(", ")]  # Clean and split
                choice_letters = [' ' + chr(65 + j) for j in range(len(choices_list))]  # Generate letter choices (' A', ' B', etc.)

                # Get the correct answer index from the dataset
                answer_idx = dataset_train['answer'][i]

                # Get the tokenized correct answer based on the index (matching with the choice letter)
                correct_choice_token = tokenizer(choice_letters[answer_idx], truncation=True, max_length=2, padding="max_length")
                correct_token_ids = correct_choice_token["input_ids"]

                # Select a random incorrect answer, ensuring it's not the correct one
                incorrect_choices = [choice for j, choice in enumerate(choices_list) if j != answer_idx]
                random_incorrect_answer = random.choice(incorrect_choices)
                random_incorrect_letter = choice_letters[choices_list.index(random_incorrect_answer)]  # Get the corresponding letter

                # Tokenize the incorrect answer (as a letter) just like the correct one
                incorrect_token_ids = tokenizer(random_incorrect_letter, truncation=True, max_length=2, padding="max_length")["input_ids"]

                # Store the correct and incorrect tokens for each sample
                correct_toks.append(correct_token_ids[1])  # Assuming tokenized choices have a space padding, we take the second token
                incorrect_toks.append(incorrect_token_ids[1])  # Same logic for the incorrect token


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

            logit_attributions = compute_logit_attribution(model, feature_acts, correct_toks, incorrect_toks)

            # Aggregate activations over positions (e.g., take max over seq_len)
            #max_feature_acts = feature_acts.abs().max(dim=1).values  # Shape: [batch_size, num_features]

            # Get sorted indices (features sorted by decreasing activation)
            sorted_indices = torch.argsort(-logit_attributions, dim=-1)  # Shape: [batch_size, num_features]
            #sorted_indices = torch.argsort(-max_feature_acts, dim=-1)  # Shape: [batch_size, num_features]

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

from functools import partial
import random

def steering_hook(
    activations,
    hook,
    steering_vector,
    steering_strength: float = 1.0,
):
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering
    vector added to it.
    """
    if steering_vector.device != activations.device:
        steering_vector = steering_vector.to(activations.device)

    return activations - steering_strength * steering_vector

def generate_with_modification(
    model,
    sae: SAE,
    input,
    feature_idx: int,
    mode: str = "baseline",
    steering_strength: float = 2.0,
    max_new_tokens: int = 5
):
    """
    Generates text with the specified modification: "baseline", "steering", "random_steering".
    """
    if mode == "baseline":
        # Baseline generation without any modification
        output = model.generate(
            input=input,
            max_new_tokens=max_new_tokens,
            **GENERATE_KWARGS
        )

    elif mode == "steering":
        # Apply specific steering logic
        _steering_hook = partial(
            steering_hook,
            steering_vector=sae.W_dec[feature_idx].sum(0)/torch.tensor(feature_idx.shape[0]).sqrt(),
            steering_strength=steering_strength,
        )
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
            output = model.generate(
                input=input,
                max_new_tokens=max_new_tokens,
                **GENERATE_KWARGS,
            )

    elif mode == "random_steering":
        # Select a random feature index for random steering
        random_feature_idx = random.randint(0, sae.W_dec.shape[0] - 1)
        random_steering_vector = sae.W_dec[random_feature_idx]

        # Apply random steering
        _random_steering_hook = partial(
            steering_hook,
            steering_vector=random_steering_vector,
            steering_strength=steering_strength,
        )
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _random_steering_hook)]):
            output = model.generate(
                input=input,
                max_new_tokens=max_new_tokens,
                **GENERATE_KWARGS,
            )

    return output


def evaluate_model_all_conditions(
    model, sae, feature_idx, dataloader, device, 
    steering_strength=1.0, max_new_tokens=5
):
    """
    Evaluates the model across three conditions: baseline, specific steering, and random steering.
    """
    model.eval()
    sae.eval()
    total_correct_baseline = 0
    total_correct_steering = 0
    total_correct_random_steering = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating all conditions"):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # Correct answer indices

        total_samples += input_ids.size(0)

        with torch.no_grad():
            # Generate outputs for baseline (no modification)
            baseline_output = generate_with_modification(
                model=model,
                sae=sae,
                input=input_ids,
                feature_idx=feature_idx,
                mode="baseline",
                max_new_tokens=max_new_tokens,
            )
            
            # Generate outputs with specific steering
            steering_output = generate_with_modification(
                model=model,
                sae=sae,
                input=input_ids,
                feature_idx=feature_idx,
                mode="steering",
                steering_strength=steering_strength,
                max_new_tokens=max_new_tokens,
            )

            # Generate outputs with random steering
            random_steering_output = generate_with_modification(
                model=model,
                sae=sae,
                input=input_ids,
                feature_idx=feature_idx,
                mode="random_steering",
                steering_strength=steering_strength,
                max_new_tokens=max_new_tokens,
            )

            # Compute token-based accuracy for all conditions
            for i in range(input_ids.size(0)):
                correct_token = labels[i][1]  # Correct token from the label

                # Baseline accuracy
                baseline_token = baseline_output[i][-2:] 
                if correct_token in baseline_token:
                    total_correct_baseline += 1

                # Specific steering accuracy
                steering_token = steering_output[i][-2:] 
                if correct_token in steering_token:
                    total_correct_steering += 1

                # Random steering accuracy
                random_steering_token = random_steering_output[i][-2:] 
                if correct_token in random_steering_token:
                    total_correct_random_steering += 1

    # Calculate accuracies for all conditions
    accuracy_baseline = total_correct_baseline / total_samples
    accuracy_steering = total_correct_steering / total_samples
    accuracy_random_steering = total_correct_random_steering / total_samples

    print(f"Accuracy (baseline): {accuracy_baseline:.4f}")
    print(f"Accuracy (specific steering): {accuracy_steering:.4f}")
    print(f"Accuracy (random steering): {accuracy_random_steering:.4f}")

    return accuracy_baseline, accuracy_steering, accuracy_random_steering


# Run evaluation for all conditions
feature_idx = idx_order[:20]  # We select the relevant features by their DLA.

accuracy_baseline, accuracy_steering, accuracy_random_steering = evaluate_model_all_conditions(
    model=model,
    sae=sae,
    feature_idx=feature_idx,
    dataloader=train_dataloader,
    device=device,
    steering_strength=2.0,  # Adjust steering strength as needed
    max_new_tokens=5        # Adjust the number of new tokens to generate as needed
)
