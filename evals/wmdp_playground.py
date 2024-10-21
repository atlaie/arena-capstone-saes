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
    release="gemma-scope-2b-pt-mlp",
    sae_id="layer_24/width_16k/average_l0_357",
    device=device
)

#%%
w_in = model.W_in[24].clone()
w_gate = model.W_gate[24].clone()
w_out = model.W_out[24].clone()

w_in.shape, w_gate.shape, w_out.shape
#%%

for i in range(len(dataset_train["choices"])):
    # Clean the choices string for the current example
    raw_choices = dataset_train["choices"][i].strip("[]")  # Remove the square brackets
    choices_list = [choice.strip("' ") for choice in raw_choices.split(", ")]  # Clean and split

    # Combine the question and the cleaned-up choices into a single input string
    combined_input = (
        dataset_train["question"][i] + "\nChoices: " + ', '.join(choices_list) + "\nAnswer:"
    )

    # Tokenize the combined input
    tokenized_inputs = tokenizer(
        combined_input,
        truncation=True,
        max_length=512
    )

    # Get the correct answer based on the index
    answer_idx = dataset_train["answer"][i]  # The index of the correct answer
    correct_answer = choices_list[answer_idx]  # Retrieve the correct answer text

    # Tokenize the correct answer (as the label for the model to predict)
    labels = tokenizer(
        correct_answer,
        truncation=True,
        max_length=512
    )["input_ids"][1]

    # Mask out padding tokens (-100 to ignore them in loss calculation)
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
#%%

choices_list, labels, correct_answer, tokenizer(
        correct_answer,
        truncation=True,
        max_length=512
    )["input_ids"][1]

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
subset_size = 64#len(tokenized_dataset_train)  # Adjust this number as needed

# Select a subset of the dataset
subset_dataset = tokenized_dataset_train.select(range(subset_size))

# Create DataLoader with the subset
batch_size = 16  # Adjust based on your GPU memory
train_dataloader = DataLoader(
    subset_dataset,
    #tokenized_dataset_train,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
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
# ====================================================================================
# ====================================================================================
# ====================================================================================
# ====================================================================================

import torch as t

# Ablation function using idx_order
def ablate_top_k_features(feature_acts, idx_order, k=20):
    # Clone the original feature activations to avoid in-place modifications
    modified_feature_acts = feature_acts.clone()
    # Get the shape of the feature activations (batch_size, seq_len, num_features)
    batch_size, seq_len, num_features = feature_acts.shape
    # Flatten the activations into (batch_size * seq_len, num_features) for easier processing
    flat_feature_acts = modified_feature_acts.view(-1, num_features)
    # Select the indices of the top-k features to ablate based on idx_order
    topk_indices = t.tensor(idx_order[:k], device=feature_acts.device)  # Top-k indices from idx_order
    # Create a mask of the same shape as flat_feature_acts, initialized to 0
    mask = torch.zeros_like(flat_feature_acts, device=feature_acts.device)
    # Apply the mask by scattering 1's into the locations of the top-k indices
    mask[:, topk_indices] = 1.0  # Ablate the specified top-k features based on idx_order
    # Zero out the top-k features by multiplying with (1 - mask)
    flat_feature_acts = flat_feature_acts * (1 - mask)
    # Reshape the modified activations back to the original shape (batch_size, seq_len, num_features)
    modified_feature_acts = flat_feature_acts.view(batch_size, seq_len, num_features)
    
    return modified_feature_acts


import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def evaluate_model_with_ablation(model, sae, tokenizer, idx_order, dataloader, device, k=20):
    model.eval()
    sae.eval()
    total_correct = 0
    total_correct_target_ablation = 0
    total_correct_random_ablation = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)
        total_samples += batch_size

        with torch.no_grad():
            # Compute logits without ablation
            logits = model(input_ids, attention_mask=attention_mask)
            logits_last_token = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

            # Get predicted tokens (the token with the highest probability)
            preds = torch.argmax(logits_last_token, dim=-1)  # Shape: [batch_size]

            # Convert the predicted token indices to actual text tokens
            predicted_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Optionally, you can also convert the true labels to text to compare them
            #true_labels = labels.view(-1)  # Flatten the labels
            # Convert the tensor to a CPU list and filter out the -100 values
            #valid_labels = true_labels[true_labels != -100].tolist()
            print(labels.shape)

            # Decode the valid labels to text
            true_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for texts in true_texts:

            # Print the decoded texts
                print(f"Decoded labels: {texts}")

            true_texts = tokenizer.batch_decode(true_labels, skip_special_tokens=True)

            # Print the predicted tokens and the true labels for comparison
            for i in range(len(predicted_texts)):
                print(f"Prediction: {predicted_texts[i]}")
                print(f"True label: {true_texts[i]}")
                print('---')


            total_correct += (preds == labels.view(-1)).sum().item()

            # Run model to get activations at the SAE layer
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=sae.cfg.hook_name,
                prepend_bos=True
            )

            activations = cache[sae.cfg.hook_name]

            # Encode activations using SAE
            feature_acts = sae.encode(activations)

            # Ablate top k active features
            target_modified_feature_acts = ablate_top_k_features(feature_acts, idx_order, k=k)
            random_modified_feature_acts = ablate_top_k_features(feature_acts, np.random.permutation(idx_order), k=k)

            # Decode back to get modified activations
            target_modified_activations = sae.decode(target_modified_feature_acts)
            random_modified_activations = sae.decode(random_modified_feature_acts)

            # Define hook function to replace activations
            def random_ablation_hook(activation, hook):
                return random_modified_activations

            def target_ablation_hook(activation, hook):
                return target_modified_activations

            # Compute logits with target ablation
            logits_ablation_target = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(sae.cfg.hook_name, target_ablation_hook)],
                return_type="logits",
            )
            logits_ablation_target_last_token = logits_ablation_target[:, -1, :]  # Shape: [batch_size, vocab_size]

            # Get predicted tokens after target ablation
            preds_target_ablation = torch.argmax(logits_ablation_target_last_token, dim=-1)  # Shape: [batch_size]
            total_correct_target_ablation += (preds_target_ablation == labels.view(-1)).sum().item()

            # Compute logits with random ablation
            logits_ablation_random = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(sae.cfg.hook_name, random_ablation_hook)],
                return_type="logits",
            )
            logits_ablation_random_last_token = logits_ablation_random[:, -1, :]  # Shape: [batch_size, vocab_size]

            # Get predicted tokens after random ablation
            preds_random_ablation = torch.argmax(logits_ablation_random_last_token, dim=-1)  # Shape: [batch_size]
            total_correct_random_ablation += (preds_random_ablation == labels.view(-1)).sum().item()

            # Clean up to free memory
            del cache, feature_acts, random_modified_feature_acts

    # Calculate accuracies
    accuracy = total_correct / total_samples
    accuracy_target_ablation = total_correct_target_ablation / total_samples
    accuracy_random_ablation = total_correct_random_ablation / total_samples

    print(f"Accuracy without ablation: {accuracy:.4f}")
    print(f"Accuracy with targeted {k}-ablation: {accuracy_target_ablation:.4f}")
    print(f"Accuracy with random {k}-ablation: {accuracy_random_ablation:.4f}")

    return accuracy, accuracy_target_ablation, accuracy_random_ablation


# Evaluation function
# from tqdm import tqdm
# import torch.nn.functional as F

# def evaluate_model_with_ablation(model, sae, idx_order, dataloader, device, k=20):
#     model.eval()
#     sae.eval()
#     total_loss = 0.0
#     total_loss_target_ablation = 0.0
#     total_loss_random_ablation = 0.0
#     total_samples = 0

#     for batch in tqdm(dataloader, desc="Evaluating"):
#         # Move data to device
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         batch_size = input_ids.size(0)
#         total_samples += batch_size

#         with torch.no_grad():
#             # Compute logits without ablation
#             logits = model(input_ids, attention_mask=attention_mask)
#             logits_last_token = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

#             # Compute loss manually
#             loss = F.cross_entropy(
#                 logits_last_token,
#                 labels.view(-1),
#                 ignore_index=-100,
#             )
#             total_loss += loss.item() * batch_size

#             # Run model to get activations at the SAE layer
#             _, cache = model.run_with_cache(
#                 input_ids,
#                 names_filter=sae.cfg.hook_name,
#                 prepend_bos=True
#             )

#             activations = cache[sae.cfg.hook_name]

#             # Encode activations using SAE
#             feature_acts = sae.encode(activations)

#             # Ablate top k active features
#             target_modified_feature_acts = ablate_top_k_features(feature_acts, idx_order, k=k)
#             random_modified_feature_acts = ablate_top_k_features(feature_acts, np.random.permutation(idx_order), k=k)

#             # Decode back to get modified activations
#             target_modified_activations = sae.decode(target_modified_feature_acts)
#             random_modified_activations = sae.decode(random_modified_feature_acts)

#             # Define hook function to replace activations
#             def random_ablation_hook(activation, hook):
#                 return random_modified_activations
#             def target_ablation_hook(activation, hook):
#                 return target_modified_activations

#             # Compute logits with ablation
#             logits_ablation_random = model.run_with_hooks(
#                 input_ids,
#                 fwd_hooks=[(sae.cfg.hook_name, random_ablation_hook)],
#                 return_type="logits",
#             )
#             # Compute logits with ablation
#             logits_ablation_target = model.run_with_hooks(
#                 input_ids,
#                 fwd_hooks=[(sae.cfg.hook_name, target_ablation_hook)],
#                 return_type="logits",
#             )
#             logits_ablation_target_last_token = logits_ablation_target[:, -1, :]  # Shape: [batch_size, vocab_size]

#             # Compute loss manually
#             loss_target_ablation = F.cross_entropy(
#                 logits_ablation_target_last_token,
#                 labels.view(-1),
#                 ignore_index=-100,
#             )

#             total_loss_target_ablation += loss_target_ablation.item() * batch_size

#             logits_ablation_target = model.run_with_hooks(
#                 input_ids,
#                 fwd_hooks=[(sae.cfg.hook_name, random_ablation_hook)],
#                 return_type="logits",
#             )
#             logits_ablation_random_last_token = logits_ablation_random[:, -1, :]  # Shape: [batch_size, vocab_size]

#             # Compute loss manually
#             loss_random_ablation = F.cross_entropy(
#                 logits_ablation_random_last_token,
#                 labels.view(-1),
#                 ignore_index=-100,
#             )

#             total_loss_random_ablation += loss_random_ablation.item() * batch_size

#             print('Original loss: ', loss)
#             print('Target ablation loss: ', loss_target_ablation)
#             print('Random ablation loss: ', loss_random_ablation)

#             del cache, feature_acts, random_modified_feature_acts

#     avg_loss = total_loss / total_samples
#     avg_loss_target_ablation = total_loss_target_ablation / total_samples
#     avg_loss_random_ablation = total_loss_random_ablation / total_samples

#     print(f"Average loss without ablation: {avg_loss}")
#     print(f"Average loss with targetted "+str(k)+"-ablation: ", avg_loss_target_ablation)
#     print(f"Average loss with random "+str(k)+"-ablation: ", avg_loss_random_ablation)

#     return avg_loss, avg_loss_ablation

# Run evaluation
k = 200  # Number of top active features to ablate
avg_loss, avg_loss_ablation = evaluate_model_with_ablation(
    model=model,
    sae=sae,
    tokenizer = tokenizer,
    idx_order = idx_order, 
    dataloader=train_dataloader,
    device=device,
    k=k
)
#%%
for batch in tqdm(train_dataloader, desc="Evaluating"):
    # Move data to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    batch_size = input_ids.size(0)

    # with torch.no_grad():
    #     # Compute logits without ablation
    #     logits = model(input_ids, attention_mask=attention_mask)
    #     logits_last_token = logits[:, -1, :]  # Shape: [batch_size, vocab_size]

    #     # Get predicted tokens (the token with the highest probability)
    #     preds = torch.argmax(logits_last_token, dim=-1)  # Shape: [batch_size]
        
    #     preds = torch.argmax(logits_last_token, dim=-1)  # Shape: [batch_size]

    #     # Convert the predicted token indices to actual text tokens
    #     predicted_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Optionally, you can also convert the true labels to text to compare them
    true_labels = labels.view(-1)  # Flatten the labels
    true_texts = tokenizer.batch_decode(true_labels, skip_special_tokens=True)

    # Print the predicted tokens and the true labels for comparison
    for i in range(len(true_texts)):
        #print(f"Prediction: {predicted_texts[i]}")
        print(f"True label: {true_texts[i]}")
        print('---')
#%%
for ii in range(5):
    print(train_data['choices'].values[ii])

#%%

input_ids_list = []
attention_mask_list = []
labels_list = []

for i in range(4):
    # Clean the choices string for the current example
    raw_choices = train_data["choices"][i].strip("[]")  # Remove the square brackets
    choices_list = [choice.strip("' ") for choice in raw_choices.split(", ")]  # Clean and split

    # Combine the question and the cleaned-up choices into a single input string
    combined_input = (
        train_data["question"][i] + "\nChoices: " + ', '.join(choices_list) + "\nAnswer:"
    )

    # Tokenize the combined input
    tokenized_inputs = tokenizer(
        combined_input,
        truncation=True,
        max_length=512
    )

    # Get the correct answer based on the index
    answer_idx = train_data["answer"][i]  # The index of the correct answer
    correct_answer = choices_list[answer_idx]  # Retrieve the correct answer text

    # Tokenize the correct answer (as the label for the model to predict)
    labels = tokenizer(
        correct_answer,
        truncation=True,
        max_length=512
    )["input_ids"]
    texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(texts)