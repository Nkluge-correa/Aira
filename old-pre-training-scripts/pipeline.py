from data_management import get_data, preprocess_data
from log import get_last_checkpoint, get_logger
from specification import get_specification
from evaluate_model import do_evaluation
from training import do_training
from model import (get_model_mlm, 
                    get_model_clm,
                    get_model_config,
                    get_tokenizer,
                    train_tokenizer,
                    get_model_card_kwargs)

import transformers
from transformers import (DataCollatorForLanguageModeling,
                            is_torch_tpu_available,
                            Trainer,
                            set_seed,)

import os
import glob
import pandas as pd
from datetime import date, datetime


def train_pipeline(
    model_type: str,
    model_name_or_path: str,
    train_from_scratch: bool,
    train_new_tokenizer: bool,
    tokenizer_vocab_size: int,
    language_modeling_task: str,
    tokenizer_name: str,
    config_name: str,
    do_train: bool,
    do_eval: bool,
    eval_steps: float,
    list_of_datasets: list,
    output_dir: str,
    max_steps: int,
    save_total_limit: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    no_cuda: bool,
    logger_name: str,
    hub_model_id: str = None,
    hub_token: str = None,
):

    model_args, list_of_data_args, training_args = get_specification( 
        model_type=model_type,
        train_from_scratch=train_from_scratch,
        train_new_tokenizer=train_new_tokenizer,
        tokenizer_vocab_size=tokenizer_vocab_size,
        language_modeling_task=language_modeling_task,
        model_name_or_path=model_name_or_path,
        tokenizer_name=tokenizer_name,
        config_name=config_name,
        do_train=do_train,
        do_eval=do_eval,
        eval_steps=eval_steps,
        list_of_datasets=list_of_datasets,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        no_cuda=no_cuda,
        max_steps=max_steps,
        hub_model_id=hub_model_id,
        hub_token=hub_token,
    )

    data_args = list_of_data_args[0]

    set_seed(training_args.seed)

    logger = get_logger(logger_name, training_args=training_args)

    logger.warning(f"Training a {model_name_or_path} model from scratch: {train_from_scratch}")

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},\n"
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        f"{bool(training_args.local_rank != -1)}, 16-bits training: "
        f"{training_args.fp16}"
    )

    last_checkpoint = get_last_checkpoint(training_args, logger=logger)

    for i in range(len(list_of_data_args)):
        logger.info(f"Loading data... {list_of_data_args[i].dataset_name}")
    
    data = get_data(list_of_data_args, data_args)

    config = get_model_config(model_args, logger=logger)

    if model_args.train_new_tokenizer:
        logger.info("Training a new tokenizer from the dataset(s)...")
        tokenizer = train_tokenizer(model_args, data, logger=logger)

    else:

        logger.info("Loading a tokenizer from the model name or path...")
        tokenizer = get_tokenizer(model_args)

    if model_args.language_modeling_task  == "clm":

        logger.warning("Loading a model for Causal Language Modeling...")
        model = get_model_clm(model_args, config=config, tokenizer=tokenizer, logger=logger)

    else:
        
        logger.warning("Loading a model for Masked Language Modeling...")
        model = get_model_mlm(model_args, config=config, tokenizer=tokenizer, logger=logger)

    logger.warning(f"MODEL IS IN {str(model.device).upper()}!")

    (
        train_dataset,
        eval_dataset,
        compute_metrics,
        preprocess_logits_for_metrics,
    ) = preprocess_data(
        data=data,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        logger=logger,
    )

    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )

    if model_args.language_modeling_task  == "clm":

        logger.warning("Using DataCollatorForLanguageModeling for CLM...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
    else:

        logger.warning("Using DataCollatorForLanguageModeling for MLM...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    logger.warning(f"""Training started at: {datetime.now().strftime("%H:%M:%S - %d/%m/%Y")}""")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    logger.info(f"MODEL IS IN {str(trainer.model.device).upper()}!")

    if training_args.do_train:
        trainer = do_training(
            trainer,
            training_args=training_args,
            train_dataset=train_dataset,
            data_args=data_args,
            last_checkpoint=last_checkpoint,
        )

    if training_args.do_eval:
        logger.warning("*** Evaluation ***")
        trainer = do_evaluation(trainer, data_args, eval_dataset)

    logger.info(f"""Training ended at: {datetime.now().strftime("%H:%M:%S - %d/%m/%Y")}""")
    
    logger.info("Creating model card...")

    kwargs = get_model_card_kwargs(model_args, data_args)
    trainer.create_model_card(**kwargs)
    
    if training_args.hub_token is not None:

        logger.info("Authentication token provided. Pushing to the Hub...")

        os.system(f"huggingface-cli login --token {hub_args.hub_token}")

        model.push_to_hub(f"{hub_args.hub_user_name}/{hub_args.hub_repo_name}-{model_args.tokenizer_name}")
        tokenizer.push_to_hub(f"{hub_args.hub_user_name}/{hub_args.hub_repo_name}-{model_args.tokenizer_name}")

    else:
        logger.info("No authentication token provided. Skipping push to the Hub.")
    
    logger.info("Done!")
