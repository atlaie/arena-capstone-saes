"""
Measuring Massive Multitask Language Understanding

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300

Based on: https://github.com/openai/simple-evals/blob/main/mmlu_eval.py

# eval all subjects w/ 500 randomly selected samples
inspect eval mmlu.py --limit 500

# add chain of thought
inspect eval mmlu.py --limit 500 -T cot=true

# eval selected subjects
inspect eval mmlu.py -T subjects=anatomy
inspect eval mmlu.py -T subjects=astronomy
inspect eval mmlu.py -T subjects=anatomy,astronomy
"""
import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


# map records to inspect sample
def record_to_sample(record):
    return Sample(
        input=record["Question"],
        choices=[
            str(record["A"]),
            str(record["B"]),
            str(record["C"]),
            str(record["D"]),
        ],
        target=record["Answer"],
        metadata={"subject": record["Subject"]},
    )


# read dataset globally so it can be shared by all of the tasks
# (shuffle so that --limit draws from multiple subjects)
dataset = csv_dataset(
    csv_file="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
    sample_fields=record_to_sample,
    shuffle=True,
)
with open('eval_config.json', 'r') as file:
        cfg = json.load(file)
if cfg['use_ratio']:
    cfg['max_sample'] = int(min(len(dataset)*cfg['ratio'], float(cfg['max_sample'])))
dataset = dataset[:max(cfg['max_sample'], cfg['min_sample'])]


@task
def mmlu(subjects=[], cot=False):
    # filter dataset if requested
    subjects = subjects if isinstance(subjects, list) else [subjects]
    if len(subjects) > 0:
        task_dataset = dataset.filter(
            name=f"{dataset.name}-{'-'.join(subjects)}",
            predicate=lambda sample: sample.metadata["subject"] in subjects,
        )
    else:
        task_dataset = dataset

    solver = multiple_choice(cot=cot)

    return Task(
        dataset=task_dataset,
        solver=solver,
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )