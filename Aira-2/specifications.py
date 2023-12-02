from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    base_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model to use for training."
            )
        },
    )

    model_ref: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reference model to used for DPO training."
            )
        },
    )

    use_fast: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use the fast method of tokenization."},
    )

    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all hidden-states (i.e., all hidden-states for all layers)."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    bos_token: Optional[str] = field(
        default='<|startofinstruction|>',
        metadata={"help": "The beginning of sentence token"},
    )

    sep_token: Optional[str] = field(
        default='<|endofinstruction|>',
        metadata={"help": "The end of sentence token"},
    )

    eos_token: Optional[str] = field(
        default='<|endofcompletion|>',
        metadata={"help": "The end of sentence token"},
    )   

    pad_token: Optional[str] = field(
        default='<|pad|>',
        metadata={"help": "The padding token"},
    )   

    unk_token: Optional[str] = field(
        default='<|unk|>',
        metadata={"help": "The unknown token"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_split: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset split to use."},
    )

    validation_split_percentage: Optional[int] = field(
        default=0.05,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )

    streaming: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enable streaming mode"}
    )

    max_prompt_length: Optional[int] = field(
        default=100,
        metadata={"help": "The maximum length of the prompt when performing DPO training."},
    )

    max_length: Optional[int] = field(
        default=400,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
            )
        },
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    pin_memory: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pin memory of the DataLoader or not."
            )
        },
    )

    sanity_check: Optional[bool] = field(
        default=False,
        metadata={"help": "If set, will run training on a small portion of the dataset."},
    )

@dataclass
class ExtraArguments:
    """
    Arguments pertaining miscellaneous things (e.g., the Accelerator, W&B, logger name, etc.).
    """
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the project."},
    )

    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use for logging to wandb."},
    )

    wandb_log_steps: Optional[int] = field(
        default=1,
        metadata={"help": "The number of steps to log to wandb."},
    )

    sample_every: Optional[int] = field(
        default=100,
        metadata={"help": "The number of steps between each time the model generates samples."},
    )

    mixed_precision: Optional[str] = field(
        default='no',
        metadata={"help": "Whether to use mixed precision or not ('no', 'fp16')."},
    )

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta value for DPO training."},
    )