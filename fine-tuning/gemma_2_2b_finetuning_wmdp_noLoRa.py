#%%
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Check if `bfloat16` is supported on our hardware (Ampere GPUs and above support this)
use_bfloat16 = is_bfloat16_supported()

# Step 1: Load and Process the sWMDP Dataset
full_synthetic_wmdp = pd.read_csv("full_synthetic_wmdp.csv")

# Split the dataset into train and test sets using a 90/10 split
DATA_SEED = 42

train_data = full_synthetic_wmdp.sample(frac=1, random_state=DATA_SEED)
train_data.reset_index(drop=True, inplace=True)

# Convert the Pandas DataFrame to Hugging Face Dataset format
dataset_train = Dataset.from_pandas(train_data)

# Step 2: Initialize the FastLanguageModel
max_seq_length = 2048  # Adjust as needed
dtype = None  # Set to None for auto detection of device capabilities
load_in_4bit = False  # Do not use 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it",
    max_seq_length=max_seq_length,
    dtype=torch.float16 if not use_bfloat16 else torch.bfloat16,  # Set correct precision
    load_in_4bit=load_in_4bit,
)
#%%
for param in model.model.layers[-1].mlp.parameters():
    print(param.shape)
#%%
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer
# for param in model.model.layers[-1].mlp.parameters():
#     param.requires_grad = True
model.model.layers[-1].mlp.parameters()[-1].requires_grad = True

# Step 3: Tokenize the Dataset
def tokenize_function_with_choices(examples):
    # Prepare the lists for batch-level tokenized outputs
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Loop over each example in the batch
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
            padding="max_length",  # Pad the sequences to max length
            truncation=True,       # Truncate sequences that are too long
            max_length=512         # Adjust based on your maximum sequence length
        )

        # Get the correct answer based on the index
        answer_idx = examples["answer"][i]  # The index of the correct answer
        correct_answer = choices_list[answer_idx]  # Retrieve the correct answer text

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

# Tokenize the training set
tokenized_dataset_train = dataset_train.map(tokenize_function_with_choices, batched=True)

#%%

# Create a list of parameters that require gradients
trainable_parameters = [p for p in model.parameters() if p.requires_grad]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset_train,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not use_bfloat16,
        bf16=use_bfloat16,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="./fine-tuned-results",
        save_safetensors=False,  # Disable safe serialization
    ),
    # Pass the trainable parameters to the optimizer
    optimizers=(torch.optim.AdamW(trainable_parameters, lr=2e-4), None),
)

# Step 4: Train the Model
trainer_stats = trainer.train()
# %%
# Untie the shared weights between embed_tokens and lm_head
import torch.nn as nn
from safetensors.torch import save_model
#model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())

# Step 5: Save the Fine-Tuned Model
save_model(model, "gemma-2-2b-it_noLoRa.safetensors")

#model.save_pretrained("gemma-2-2b-it_noLoRa")
#tokenizer.save_model("gemma-2-2b-it_noLoRa")

# Output training statistics
print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
# %%

from transformers import AutoModelForCausalLM

model_base = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it').to('cuda:0')

model_ft = AutoModelForCausalLM.from_pretrained('./gemma-2-2b-it-wmdp/gemma-2-2b-it-noQuant').to('cuda:0')

#%%
import torch as t

t.sum(model_base.model.layers[-1].mlp.down_proj.weight - model_ft.model.layers[-1].mlp.down_proj.weight)

#%%
import seaborn as sns

mean_weights_ft = model_ft.model.layers[-1].mlp.down_proj.weight.to('cpu').detach().numpy()
mean_weights_base = model_base.model.layers[-1].mlp.down_proj.weight.to('cpu').detach().numpy()
sns.heatmap(mean_weights_ft - mean_weights_base)
model_base.model.layers[-1].mlp.down_proj.weight.shape


#%%
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

from datasets import load_dataset
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import FastLanguageModel, is_bfloat16_supported

# Check if `bfloat16` is supported on our hardware (Ampere GPUs and above support this)
use_bfloat16 = is_bfloat16_supported()

# Step 2: Initialize the FastLanguageModel
max_seq_length = 2048  # Adjust as needed
dtype = None  # Set to None for auto detection of device capabilities
load_in_4bit = False  # Do not use 4-bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it",
    max_seq_length=max_seq_length,
    dtype=torch.float16 if not use_bfloat16 else torch.bfloat16,  # Set correct precision
    load_in_4bit=load_in_4bit,
)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

for ii, elem in enumerate(model.model.layers[-1].mlp.parameters()):
    if ii == 2:
        elem.requires_grad = True

# Create a list of parameters that require gradients
trainable_parameters = [p for p in model.parameters() if p.requires_grad]

# Load the dataset
ds = load_dataset("cais/wmdp", "wmdp-bio")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            padding = 'max_length',
            max_length=512,
        )

        # Tokenize the prompt to find its length
        tokenized_prompt = tokenizer(
            combined_input,
            truncation=True,
            max_length=512,
            padding = 'max_length',
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

batch_size = 32  # Adjust based on your GPU memory
test_dataloader = DataLoader(
    tokenized_dataset_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset_test,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not use_bfloat16,
        bf16=use_bfloat16,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="./noLoRa-original-dataset-results",
        save_safetensors=False,  # Disable safe serialization
    ),
    # Pass the trainable parameters to the optimizer
    optimizers=(torch.optim.AdamW(trainable_parameters, lr=2e-4), None),
)

# Step 4: Train the Model
trainer_stats = trainer.train()

#%%

model.save_pretrained("./gemma-2-2b-it_noLoRa_original-dataset", safe_serialization=False)
#tokenizer.save_pretrained("./gemma-2-2b-it_noLoRa_original-dataset", safe_serialization=False)
##
#%%

from transformers import AutoModelForCausalLM
model_noLora = AutoModelForCausalLM.from_pretrained('gemma-2-2b-it_noLoRa_original-dataset')