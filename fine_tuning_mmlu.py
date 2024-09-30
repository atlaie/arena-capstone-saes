# %%
# Install necessary libraries
!pip install -U accelerate bitsandbytes datasets peft transformers
!pip install huggingface_hub
!pip install -q -U trl
!pip install -U optimum

# %%
# Import necessary libraries
import nltk; nltk.download('punkt')
#from huggingface_hub import notebook_login
#notebook_login()

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, set_seed
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, setup_chat_format
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator
import logging
import transformers
import datasets
import sys
import os
from datasets import Dataset, load_dataset  # Import load_dataset to load MMLU
import pandas as pd

logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

# Define the output directory
output_dir = "./results-mmlu"

# Choose your model
model_id = "EleutherAI/gpt-neo-125m"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage="uint8",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="main",
    trust_remote_code=False,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto",
    quantization_config=quantization_config,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision="main",
    trust_remote_code=False,
    add_eos_token=True,
    use_fast=True
)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048

# %%
# Load the MMLU dataset
subject = 'all'  # You can specify a subject or use 'all' to include all subjects
dataset = load_dataset("hendrycks_test", subject)

# Preprocess the dataset to create 'text' field
def preprocess_function(examples):
    questions = examples['question']
    choices = examples['choices']
    answers = examples['answer']

    texts = []
    for question, choice, answer in zip(questions, choices, answers):
        choice_texts = choice['text']  # List of choices, e.g., ['A) ...', 'B) ...', ...]
        text = "Question: {}\nChoices:\n{}\nAnswer: {}".format(
            question,
            "\n".join(choice_texts),
            answer
        )
        texts.append(text)
    return {'text': texts}

# Apply the preprocessing function to the dataset
processed_dataset = dataset.map(preprocess_function, batched=True)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

# Prepare training and evaluation datasets
tokenized_dataset_training = tokenized_dataset['train']
tokenized_dataset_test = tokenized_dataset['test']

# %%
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = "INFO"
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

last_checkpoint = None
if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint is not None:
    logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

# %%
# Prepare the model for k-bit training and configure PEFT
model = prepare_model_for_kbit_training(model)
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

# Set up the training arguments (adjusted if necessary)
training_args = TrainingArguments(
    output_dir="./results-mmlu",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    log_level="debug",
    logging_steps=10,
    num_train_epochs=3,  # Adjusted to use epochs instead of max_steps
    save_strategy="epoch",
    warmup_steps=25,
    lr_scheduler_type="linear"
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset_training,
    eval_dataset=tokenized_dataset_test,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Start training
trainer.train()

# %%
# Save the fine-tuned model
model.save_pretrained("./fine-tuned-mmlu")
tokenizer.save_pretrained("./fine-tuned-mmlu")
