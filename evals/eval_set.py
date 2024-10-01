# %%
from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs

success, logs = eval_set(
   tasks=["wmdp-fast.py","arc.py","gpqa.py"],
   model=["hf/local"],
   model_args={"model_path": "../gemma-2-2b-it"},
   log_dir="logs-run-test",
   max_connections=1
)

if success:
    results = list_eval_logs("logs-run-test")
    print(results)
# %%
