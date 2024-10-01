# %%
from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs
from inspect_ai.solver import multiple_choice

success, logs = eval_set(
#    tasks=["gpqa.py"],
   tasks=["wmdp.py","arc.py","gpqa.py","mmlu.py","mmlu-pro.py"],
   model=["hf/local"],
   model_args={"model_path": "../gemma-2-2b-it"},
   log_dir="logs-run-test7",
   max_connections=2
)

if success:
    results = list_eval_logs("logs-run-test")
    print(results)
# %%
