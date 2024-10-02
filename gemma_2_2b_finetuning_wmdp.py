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

# Add LoRA adapters for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Number of LoRA ranks
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,  # Dropout for LoRA
    bias="none",       # Bias term configuration
    use_gradient_checkpointing="unsloth",  # Memory optimization
    random_state=3407,
    use_rslora=False  # Rank stabilized LoRA off
)

#print(model.named_modules)
print(model.base_model.model.model.layers[-1].mlp)

#%%

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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset_train,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not use_bfloat16,  # Use float16 if bfloat16 is not supported
        bf16=use_bfloat16,  # Enable bfloat16 if supported
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="./fine-tuned-results",
    ),
)

# Step 4: Train the Model
trainer_stats = trainer.train()

print(model.named_modules)

# Output training statistics
print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
# %%
# Step 5: Save the Fine-Tuned Model
model.save_pretrained("./gemma-2-2b-it_full-precision")
tokenizer.save_pretrained("./gemma-2-2b-it_full-precision")
# %%


# ====================================================================================


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

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer (e.g., 'lm_head')
for param in model.model.layers[-1].parameters():
    param.requires_grad = True

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

#model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())

# Step 5: Save the Fine-Tuned Model
model.save_pretrained("gemma-2-2b-it_noLoRa")
#tokenizer.save_model("gemma-2-2b-it_noLoRa")

# Output training statistics
print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds.")
# %%

from transformers import AutoModelForCausalLM

model_base = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it').to('cuda:0')

#model_ft = AutoModelForCausalLM.from_pretrained('./gemma-2-2b-it-wmdp')

#%%
import torch as t



#%%

