# Unlearning safety fine-tuning through SAE latent steering

This repository contains the code and data for the **ARENA 4.0** capstone project. 

## Repository Structure

- `evals/`: Contains the model evaluation we wrote using AISI's Inspect.
- `fine-tuning/`: Contains the relevant scripts for model fine-tuning. We tried different approaches (LoRa, PEFT, manually freezing weights)
- `results/`: Contains relevant plots for the project.
- `wandb/`: Contains model fine-tuning runs, for reproducibility.

We specify all library dependencies on `requirements.txt`.

## Overview

We first fine-tune Gemma-2-2b-it on a synthetic dataset (`full_synthetic_wmdp.csv`) to have more dangerous knowledge (chemical/bio-weapons, cyber-attacks, etc).
Then, we perform Direct Logit Attribution over layers and find the most promising one that we'll target (layer 24, in this case).
For that layer, we use the pre-trained 16-k wide SAE, coming from the recent release by Google Deepmind.
We select the most active SAE latents (`SAE_steering.py`) in the synthetic dataset (selected through a ranking analysis) and ablate their projections to the model's residual stream. We uploaded the ablated model on HuggingFace (http://huggingface.co/atlaie/gemma-2-2b-it-wmdp/tree/main).
We run the evaluation (Weapons of Mass Destruction Proxy dataset).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
