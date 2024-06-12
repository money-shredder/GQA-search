# GQA Search

GQA Search on T5-small model. 

Train GQA of T5-small:
```
python train_gqa_t5.py 

```
Train Lora of T5-small:
```
python train_lora_t5.py 

```
Train GQA-Lora of T5-small:
```
python train_gqa_lora_t5.py 

```

Parameters: 

* `Seach type`: Modify `randomized_search.py`  (Line 442 in [randomized_search.py] `find_asym_grouping` or `find_grouping`).
* `Seach strategies`: Modify `distance`  (Line 419 in [train_gqa_t5.py]) or `grouping_metrics.py`.
* `Other parameters`: Including `model_config` (Change to T5-XXL may need to modify `configuration_t5.py` and `configuration_t5_lora.py`), `path`, `Hyperparameters` (Lora and GQA)

The above instructions apply only to GLUE task (SST2, QNLI, MNLI).

For MMLU task:

```
train eval_prompting.py
```

TODO: Lora And GQA-Lora

