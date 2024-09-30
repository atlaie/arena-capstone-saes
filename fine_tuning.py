# %%

!pip install -U bitsandbytes peft transformers
!pip install -q -U trl
!pip install -U optimum

!pip install convokit

# %%

import nltk; nltk.download('punkt')
from convokit import Corpus, download

# Load the Diplomacy corpus
corpus = Corpus(filename=download('diplomacy-corpus'))
corpus_train = Corpus(filename=download('diplomacy-corpus'))
corpus_train.filter_conversations_by(lambda convo: convo.meta.get('acl2020_fold')=='Train')
corpus_train.print_summary_stats()

# Extract data where deception was both intended and effective
effective_deceptive_data_training = []
for conv in corpus_train.iter_conversations():
    for utt in conv.iter_utterances():
        if utt.meta['speaker_intention'] == 'Lie' and utt.meta['receiver_perception'] == 'Truth':
            effective_deceptive_data_training.append({
                "text": utt.text,
                "speaker": utt.speaker.id,
                "conversation_id": conv.id,
            })

# %%

from huggingface_hub import notebook_login
notebook_login()

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
import pandas as pd
from datasets import Dataset


logger = logging.getLogger(__name__)


# Set seed for reproducibility
set_seed(42)

# Define the output directory
output_dir = "./results-deception"

#model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_id="google/gemma-2-2b-it"
model_id="EleutherAI/gpt-neo-125m"
#model_id="microsoft/phi-1"


quantization_config = BitsAndBytesConfig(
    #load_in_4bit=True,
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
    # use_flash_attention_2=True,
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



# Tokenize the dataset_train
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Convert the effective deceptive data into a DataFrame
df_training = pd.DataFrame(effective_deceptive_data_training)
#df_test = pd.DataFrame(effective_deceptive_data_test)

# Create a Dataset object
dataset_train = Dataset.from_pandas(df_training)
#dataset_test = Dataset.from_pandas(df_test)

tokenized_dataset_training = dataset_train.map(tokenize_function, batched=True)
#tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True)

# %%

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

model = prepare_model_for_kbit_training(model)
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results-deception",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    log_level="debug",
    logging_steps=10,
    eval_steps=25,
    max_steps=75,
    save_steps=25,
    warmup_steps=25,
    lr_scheduler_type="linear"
)


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
trainer.train()


# %%

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-diplomacy-deception-phi1")
tokenizer.save_pretrained("./fine-tuned-diplomacy-deception-phi1")