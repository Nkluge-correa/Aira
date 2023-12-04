import os
import sys
import yaml
import torch
import wandb
import logging
import argparse
from datasets import load_dataset
from huggingface_hub import create_repo, HfApi

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from trl import RewardTrainer

from accelerate.logging import get_logger

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

def main(spec_file):

    # Load the arguments from the spec file
    with open(spec_file, "r") as stream:
        all_kwargs = yaml.safe_load(stream)

    # Get the arguments for the model, data, training, and extra arguments (wandb, etc.) 
    model_args = ModelArguments(**all_kwargs['model_args'])
    data_args = DataTrainingArguments(**all_kwargs['data_args'])
    training_args = TrainingArguments(**all_kwargs['training_args'])
    extra_args = ExtraArguments(**all_kwargs['extra_args'])

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Create a HuggingFace repository if needed
    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is not None:
            create_repo(
                repo_id=training_args.hub_model_id, 
                token=training_args.hub_token,
                repo_type="model",
                exist_ok=True,
                private=True)
        
        else:
            raise ValueError("No model id provided. Try running with `hub_model_id=your-user-name/your-model-name`")

    # Set the logger
    logger = get_logger(extra_args.project_name)

    # Create configurations for the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load the fine-tuning dataset
    if data_args.dataset_name is not None:

        dataset = load_dataset(
            data_args.dataset_name, 
            split=data_args.dataset_split,
            use_auth_token=training_args.hub_token if training_args.hub_token else None,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )          

        # Sanity check: use only the first 100 examples
        if data_args.sanity_check:
            dataset = dataset.select(range(100))

            logger.info(f"Sanity check: using only the first 100 examples")
    
    else:

        raise ValueError("No dataset provided. Try running with `dataset_name=nicholasKluge/reward-aira-dataset`")
    
    # Load the tokenizer and model
    if model_args.base_model is not None:

        model = AutoModelForSequenceClassification.from_pretrained(model_args.base_model, num_labels=1)
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        
        logger.info(f"Model to train (base architecture): {model_args.base_model}")

    else:

        raise ValueError("No base model provided. Try running with `base_model=bert-base-cased`")
    
    # Preprocess the dataset
    def preprocess(examples):
        kwargs = {"padding": "max_length", "truncation": True, "max_length": data_args.max_length, "return_tensors": "pt"}

        prompt_plus_chosen_response = examples["instruction"] + tokenizer.sep_token + examples["chosen_response"]
        prompt_plus_rejected_response = examples["instruction"] + tokenizer.sep_token + examples["rejected_response"]

        # Then tokenize these modified fields.
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
        }
    
    # Apply preprocessing function on the dataset
    formatted_dataset = dataset.map(preprocess)

    if training_args.do_eval:
        formatted_dataset = formatted_dataset.train_test_split(test_size=data_args.validation_split_percentage)

        logger.info(f"Train set size: {len(formatted_dataset['train']):,} | Validation set size: {len(formatted_dataset['test']):,}")
    
    else:
        logger.info(f"Train set size: {len(formatted_dataset):,}")

    # Initialize W&B tracker if needed
    if extra_args.wandb_token is not None: 
        # Login to wandb    
        wandb.login(key=extra_args.wandb_token)

        # Initialize wandb
        wandb.init(
            project=extra_args.project_name, 
            notes="Fine tuning base model on the AIRA-reward dataset",
            tags=["Alignment", "reward-modeling", "Aira"],
            config=all_kwargs
        )

    # Set up the training arguments
    train_args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        do_eval=training_args.do_eval,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size if training_args.do_eval else None,
        evaluation_strategy=training_args.evaluation_strategy if training_args.do_eval else "no",
        save_strategy=training_args.save_strategy,
        logging_strategy=training_args.logging_strategy,
        logging_steps=training_args.logging_steps,
        max_steps=training_args.max_steps,
        save_steps=training_args.save_steps,
        learning_rate=training_args.learning_rate,
        report_to=['wandb', 'codecarbon'] if extra_args.wandb_token is not None else ['codecarbon'],
    )

    # Set up the Rewardtrainer
    trainer = RewardTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset if not training_args.do_eval else formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"] if training_args.do_eval else None,
    )

    # Train the model
    trainer.train()
    logger.info("Training complete!")

    # Resume wandb tracking
    if extra_args.wandb_token is not None:
        wandb.finish()

    # Push the model checkpoint to the hub if needed
    if training_args.push_to_hub and training_args.hub_token is not None:

        logger.info(f"""Ouput directory (`{os.path.join(training_args.output_dir, f"checkpoint-{training_args.max_steps}")}`) being uploaded to the hub.""")

        api = HfApi(
            token=training_args.hub_token,
        )

        api.upload_folder(
            repo_id=training_args.hub_model_id,
            folder_path=os.path.join(training_args.output_dir, f"checkpoint-{training_args.max_steps}"),
        )

        api.upload_file(
            path_or_fileobj=f"./{training_args.output_dir}/emissions.csv",
            path_in_repo=f"emissions.csv",
            repo_id=training_args.hub_model_id,
        )
        
        logger.info(f"""Output directory (`{os.path.join(training_args.output_dir, f"checkpoint-{training_args.max_steps}")}`) uploaded to the hub!""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune a language model on the Aira reward dataset.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python reward-modeling.py --spec-file reward-modeling-specs.yaml


