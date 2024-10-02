#%%

from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def wmdp_task(dataset_name):
    return Task(
        dataset=hf_dataset(
            path="cais/wmdp",
            name=dataset_name,
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )


@task
def wmdp():
    return wmdp_task("wmdp-bio")


def record_to_sample(record):
    # The choices are already a list in the record
    choices = record["choices"]
    
    # The answer is an integer index
    answer_idx = record['answer']
    
    # Convert the answer index to a letter (A, B, C, D)
    target = chr(ord('A') + answer_idx)
    
    # The question is already provided in the record
    input_question = record['question']
    
    # Return sample
    return Sample(
        input=input_question,
        choices=choices,
        target=target
    )
#%%
import os 
#os.environ["INSPECT_LOG_DIR"] = "../logs_gemma/"
logs = eval(wmdp_task, model="hf/local", 
            model_args=dict(model_path="../google/gemma-2-2b-it"))

#%%
from inspect_ai.model import get_model

model = get_model("hf/local/google/gemma-2-2b-it", device="cuda:0")
#%%
# ====================================================================================
# ====================================================================================
# ====================================================================================
# ====================================================================================

from datasets import load_dataset
from sae_lens import SAE
import torch as t
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

ds = load_dataset("cais/wmdp", "wmdp-bio")
device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained("google/gemma-2-2b-it", device = device, local_files_only = True)
tokenizer = AutoTokenizer.from_pretrained('./google/gemma-2-2b-it') 

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = "layer_25/width_16k/canonical",
    device = device
)

#%%
def tokenize_function_with_choices_test(examples):
    # Prepare the lists for batch-level tokenized outputs
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Loop over each example in the batch
    for i in range(len(examples["choices"])):
        # Combine the question and the cleaned-up choices into a single input string
        combined_input = (
            examples["question"][i] + "\nChoices: " + ', '.join(examples["choices"][i]) + "\nAnswer:"
        )

        # Tokenize the combined input
        tokenized_inputs = tokenizer(
            combined_input,
            padding="max_length",  # Pad the sequences to max length
            truncation=True,       # Truncate sequences that are too long
            max_length=512         # Adjust based on your maximum sequence length
        )

        # Get the correct answer based on the index
        answer_idx = examples["answer"][i]  # The index of the correct answer
        correct_answer = examples["choices"][i][answer_idx]  # Retrieve the correct answer text

        # Tokenize the correct answer (as the label for the model to predict)
        labels = tokenizer(
            correct_answer,
            padding="max_length",
            truncation=True,
            max_length=512  # Adjust based on your maximum label length
        )["input_ids"]

        # Mask out padding tokens (-100 to ignore them in loss calculation)
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]

        # Append the tokenized data to the respective lists
        input_ids_list.append(tokenized_inputs["input_ids"])
        attention_mask_list.append(tokenized_inputs["attention_mask"])
        labels_list.append(labels)

    # Return a dictionary with keys for 'input_ids', 'attention_mask', and 'labels'
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# Tokenize the test set
tokenized_dataset_test = ds['test'].map(tokenize_function_with_choices_test, batched=True)

#%%
import plotly.express as px

t.set_grad_enabled(False)

sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with t.no_grad():
    # activation store can give us tokens.
    batch_tokens = t.tensor(tokenized_dataset_test[:6]["input_ids"], device = device)
    _, cache = model.run_with_cache(batch_tokens, 
    names_filter = sae.cfg.hook_name,
    prepend_bos=True)

    # Use the SAE
    feature_acts = sae.encode(cache[sae.cfg.hook_name])
    sae_out = sae.decode(feature_acts)

    # save some room
    del cache

    # ignore the bos token, get the number of features that activated in each token, averaged accross batch and position
    l25 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()
    print("average l25", l25.mean().item())
    px.histogram(l25.flatten().cpu().numpy()).show()

#%%


#%%

from transformer_lens import utils
from functools import partial

# next we want to do a reconstruction test.
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return t.zeros_like(activation)

print("Orig", model(batch_tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[
            (
                sae.cfg.hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
    ).item(),
)









#%%
# ====================================================================================
# ====================================================================================
# ====================================================================================
# ====================================================================================



from datasets import load_dataset
from sae_lens import SAE
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# Load the dataset
ds = load_dataset("cais/wmdp", "wmdp-bio")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = HookedTransformer.from_pretrained(
    "google/gemma-2-2b-it", device=device, local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained('./google/gemma-2-2b-it')

# Load the SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_25/width_16k/canonical",
    device=device
)

#%%
# Tokenization function
def tokenize_function_with_choices_test(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for question, choices, answer_idx in zip(examples["question"], examples["choices"], examples["answer"]):
        # Construct the prompt
        combined_input = question + "\nChoices: " + ', '.join(choices) + "\nAnswer:"
        # Construct the target
        correct_answer = choices[answer_idx]

        # Concatenate prompt and target
        full_text = combined_input + " " + correct_answer

        # Tokenize the full text
        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
        )

        # Tokenize the prompt to find its length
        tokenized_prompt = tokenizer(
            combined_input,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        )

        # Create labels by copying input_ids
        input_ids = tokenized_full["input_ids"]
        labels_ids = input_ids.copy()

        # Mask out the tokens corresponding to the prompt
        prompt_length = len(tokenized_prompt["input_ids"])
        labels_ids[:prompt_length] = [-100] * prompt_length  # Set prompt tokens to -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(tokenized_full["attention_mask"])
        labels_list.append(labels_ids)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


# Tokenize the test set
tokenized_dataset_test = ds['test'].map(
    tokenize_function_with_choices_test,
    batched=True,
    remove_columns=ds['test'].column_names
)

# Initialize the Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    return_tensors="pt",
    label_pad_token_id=-100
)

# Create DataLoader
from torch.utils.data import DataLoader

batch_size = 16  # Adjust based on your GPU memory
test_dataloader = DataLoader(
    tokenized_dataset_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

# Ablation function
def ablate_top_k_features(feature_acts, k=1):
    modified_feature_acts = feature_acts.clone()
    batch_size, seq_len, num_features = feature_acts.shape
    flat_feature_acts = modified_feature_acts.view(-1, num_features)
    topk_values, topk_indices = flat_feature_acts.topk(k, dim=-1)
    mask = torch.zeros_like(flat_feature_acts, device=feature_acts.device)
    mask.scatter_(1, topk_indices, 1.0)
    flat_feature_acts = flat_feature_acts * (1 - mask)
    modified_feature_acts = flat_feature_acts.view(batch_size, seq_len, num_features)
    return modified_feature_acts

# Evaluation function
from tqdm import tqdm
import torch.nn.functional as F

def evaluate_model_with_ablation(model, sae, dataloader, device, k=1):
    model.eval()
    sae.eval()
    total_loss = 0.0
    total_loss_ablation = 0.0
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

            # Compute loss manually
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * batch_size

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
            modified_feature_acts = ablate_top_k_features(feature_acts, k=k)

            # Decode back to get modified activations
            modified_activations = sae.decode(modified_feature_acts)

            # Define hook function to replace activations
            def ablation_hook(activation, hook):
                return modified_activations

            # Compute logits with ablation
            logits_ablation = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(sae.cfg.hook_name, ablation_hook)],
                return_type="logits",
            )

            # Compute loss with ablation manually
            loss_ablation = F.cross_entropy(
                logits_ablation.view(-1, logits_ablation.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss_ablation += loss_ablation.item() * batch_size
            print('Original loss: ', loss)
            print('Ablation loss: ', loss_ablation)

            del cache, feature_acts, modified_feature_acts

    avg_loss = total_loss / total_samples
    avg_loss_ablation = total_loss_ablation / total_samples

    print(f"Average loss without ablation: {avg_loss}")
    print(f"Average loss with 10-ablation: {avg_loss_ablation}")

    return avg_loss, avg_loss_ablation

# Run evaluation
k = 10  # Number of top active features to ablate
avg_loss, avg_loss_ablation = evaluate_model_with_ablation(
    model=model,
    sae=sae,
    dataloader=test_dataloader,
    device=device,
    k=k
)
