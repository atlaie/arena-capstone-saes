"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord
https://arxiv.org/abs/1803.05457

# run all subsets
inspect eval arc.py

# run specific subsets
inspect eval arc.py@arc_easy
inspect eval arc.py@arc_challenge
"""

from inspect_ai import Task, task
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
