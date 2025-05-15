from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import BatchSamplers

def get_training_args(
    ckpt_dir: str,
    batch_size_gpl: int,
    use_amp: bool,
    training_args_kwargs: dict = None
) -> SentenceTransformerTrainingArguments:
    
    # Default arguments
    default_args = {
        "output_dir": ckpt_dir,
        "num_train_epochs": 2,
        "per_device_train_batch_size": batch_size_gpl,
        "per_device_eval_batch_size": 8,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "fp16": use_amp,
        "bf16": False,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "gradient_checkpointing": True,
        "report_to": None,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 100,
    }

    # If overrides are provided, update the default_args
    if training_args_kwargs:
        default_args.update(training_args_kwargs)

    return SentenceTransformerTrainingArguments(**default_args)