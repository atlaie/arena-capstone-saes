# %%

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Based on: https://github.com/openai/simple-evals/blob/main/gpqa_eval.py

# eval for default epochs (4)
inspect eval gpqa.py

# eval with 1 epoch
inspect eval gpqa.py --epochs 1

# without chain of thought
inspect eval gpqa.py -T cot=false
"""
import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

# default epochs to run eval for
DEFAULT_EPOCHS = 1


# map records to inspect samples (note that target is always "A" in the,
# dataset, we will shuffle the presentation of options to mitigate this)
def record_to_sample(record):
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )





@task
def gpqa_diamond(cot=True):
    dataset = csv_dataset(csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
            sample_fields=record_to_sample, shuffle=True)
    with open('eval_config.json', 'r') as file:
        cfg = json.load(file)
    if cfg['use_ratio']:
        cfg['max_sample'] = int(min(len(dataset)*cfg['ratio'], float(cfg['max_sample'])))
    dataset = dataset[:max(cfg['max_sample'], cfg['min_sample'])]

    return Task(
        dataset=dataset,
        solver=[
            multiple_choice(shuffle=True),
        ],
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
        epochs=DEFAULT_EPOCHS,
    )
# %%
