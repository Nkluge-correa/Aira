from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (AutoConfig,
                            AutoModelForMaskedLM,
                            AutoModelForCausalLM,
                            AutoTokenizer,
                            CONFIG_MAPPING)

def get_model_mlm(model_args, config, tokenizer, logger):

    if model_args.train_from_scratch:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
        
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model.resize_token_embeddings(len(tokenizer))
    return model

def get_model_clm(model_args, config, tokenizer, logger):
    
    if model_args.train_from_scratch:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_model_config(model_args, logger):
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.info("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            print(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
            print(f"New config: {config}")

    return config


def get_tokenizer(model_args):

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script. You can do it from another script, save it, and load it "
            "from here, using --tokenizer_name."
        )
    
    if model_args.language_modeling_task  == "clm":
        new_tokenizer.pad_token = new_tokenizer.eos_token

    return tokenizer

def train_tokenizer(model_args, data, logger):

    tokenizer_dataset = concatenate_datasets([data['train'], data['test']])
    tokenizer_dataset = tokenizer_dataset.remove_columns([col for col in tokenizer_dataset.column_names if col != "text"])
    
    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(tokenizer_dataset), batch_size)):
            yield tokenizer_dataset[i : i + batch_size]["text"]

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    new_tokenizer = tokenizer.train_new_from_iterator(
        text_iterator=batch_iterator(), 
        vocab_size=len(tokenizer.get_vocab()) if model_args.tokenizer_vocab_size is None else model_args.tokenizer_vocab_size)
    
    if model_args.language_modeling_task  == "clm":
        new_tokenizer.pad_token = new_tokenizer.eos_token

    logger.info(f"New tokenizer inherits from {new_tokenizer.__class__}")
    logger.info(f"New tokenizer vocab size: {len(new_tokenizer)}")

    return new_tokenizer


def get_model_card_kwargs(model_args, data_args):

    if model_args.language_modeling_task  == "clm":
        task = "text-generation"
    else:
        task = "fill-mask"
        
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": task}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    return kwargs
