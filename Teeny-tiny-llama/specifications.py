from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_to_train: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. "
                "Models that have the same embedding size as the default one (768), are: "
                "`EleutherAI/pythia-160m-deduped`, `gpt2`, and `distilgpt2`."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path."},
    )

    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether to train the model from scratch."},
    )

    vocab_size: Optional[int] = field(
        default=32000,
        metadata={"help": "The vocab size of the tokenizer."},
    )

    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "The hidden size of the model. Only used if `train_from_scratch` is set to `True`."},
    )

    intermediate_size: Optional[int] = field(
        default=3072,
        metadata={"help": "The intermediate size of the model. Only used if `train_from_scratch` is set to `True`."},
    )

    max_position_embeddings: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum sequence length that this model might ever be used with. Only used if `train_from_scratch` is set to `True`."},
    )

    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "The number of attention heads used by the model. Only used if `train_from_scratch` is set to `True`."},
    )

    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of hidden layers used by the model. Only used if `train_from_scratch` is set to `True`."},
    )

    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to return all hidden-states (i.e., all hidden-states for all layers)."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
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

    use_cache: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to cache the loaded pretrained weights. Set to `False` to avoid caching when loading a model."
            )
        },
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

    dataset_is_tokenized: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the dataset is already tokenized or not."},
    )

    dataset_split: Optional[str] = field(
        default="train",
        metadata={"help": "The dataset split to use."},
    )

    validation_split_percentage: Optional[int] = field(
        default=0.01,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )

    streaming: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enable streaming mode"}
    )

    block_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    overwrite_cache: Optional[bool] = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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
    wandb_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use for logging to wandb."},
    )

    logger_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the logger to use."},
    )

    wandb_log_steps: Optional[int] = field(
        default=1,
        metadata={"help": "The number of steps to log to wandb."},
    )

    with_tracking: Optional[bool] = field(
        default=True,
        metadata={"help": "Let the Accelerator track the model and optimizer states."},
    )

    sample_every: Optional[int] = field(
        default=100,
        metadata={"help": "The number of steps between each time the model generates samples."},
    )

    mixed_precision: Optional[str] = field(
        default='no',
        metadata={"help": "Whether to use mixed precision or not ('no', 'fp16', `bf16`)."},
    )

    checkpointing_steps: Optional[int] = field(
        default=None,
        metadata={"help": "The number of steps the various states should be saved at the end of every n steps."},
    )

    generation_seeds: Optional[list] = field(
        default=None,
        metadata={"help": "The generation seeds to use."},
    )