#%%

from sae_lens import SAE
import torch as t
from transformer_lens import HookedTransformer

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained("google/gemma-2-2b", device = device, local_files_only = True)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res-canonical",
    sae_id = "layer_25/width_16k/canonical",
    device = device
)
#%%

# import os
# from unsloth import FastLanguageModel, is_bfloat16_supported


# use_bfloat16 = is_bfloat16_supported()
# max_seq_length = 2048  # Adjust as needed
# dtype = None  # Set to None for auto detection of device capabilities
# load_in_4bit = True  # Use 4-bit quantization

# #os.environ['HF_HUB_HOME'] = './gemma-2-2b-lora_model'
# # model = transformer_lens.HookedTransformer.from_pretrained('adapter_model')


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="",
#     max_seq_length=max_seq_length,
#     dtype=t.float16 if not use_bfloat16 else t.bfloat16,  # Set correct precision
#     load_in_4bit=load_in_4bit,
# )

# hooked_model = HookedTransformer.from_pretrained("gemma-2-2b", hf_model=model)

#%%
import plotly.express as px
import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens.utils import tokenize_and_concatenate

dataset = load_dataset(
    path = "NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset= dataset,# type: ignore
    tokenizer = model.tokenizer, # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

t.set_grad_enabled(False)

#%%
sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

with t.no_grad():
    # activation store can give us tokens.
    batch_tokens = token_dataset[:5]["tokens"]
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

example_prompt = "When John and Mary went to the shops, John gave the bag to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

logits, cache = model.run_with_cache(example_prompt, prepend_bos=True)
tokens = model.to_tokens(example_prompt)
sae_out = sae(cache[sae.cfg.hook_name])


def reconstr_hook(activations, hook, sae_out):
    return sae_out


def zero_abl_hook(mlp_out, hook):
    return t.zeros_like(mlp_out)


hook_name = sae.cfg.hook_name

print("Orig", model(tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_abl_hook)],
    ).item(),
)


with model.hooks(
    fwd_hooks=[
        (
            hook_name,
            partial(reconstr_hook, sae_out=sae_out),
        )
    ]
):
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)