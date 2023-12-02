import os
import yaml
import torch
import logging
import argparse
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo, HfApi

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from trl import DPOTrainer

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

def main(spec_file):

    # Load the arguments from the spec file
    with open(spec_file, "r") as stream:
        all_kwargs = yaml.safe_load(stream)

    # Get the arguments for the model, data, training, and extra arguments (wandb, DPO arguments, etc.) 
    model_args = ModelArguments(**all_kwargs['model_args'])
    data_args = DataTrainingArguments(**all_kwargs['data_args'])
    training_args = TrainingArguments(**all_kwargs['training_args'])
    extra_args = ExtraArguments(**all_kwargs['extra_args'])

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Create a HuggingFace repository if needed
    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is None:
            training_args.hub_model_id = create_repo(
                repo_id=extra_args.project_name, 
                token=training_args.hub_token,
                repo_type="model",
                exist_ok=True,
                private=True)['id']
        
        else:
            create_repo(
                repo_id=training_args.hub_model_id, 
                token=training_args.hub_token,
                repo_type="model",
                exist_ok=True,
                private=True)

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
    
    # Load the tokenizer, the model, and the reference model
    if model_args.base_model is not None:

        model = AutoModelForCausalLM.from_pretrained(model_args.base_model)
        model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_ref)
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

        model.config.use_cache = False
        
        logger.info(f"Model to train (base architecture): {model_args.base_model}")

    else:

        raise ValueError("No base model provided. Try running with `base_model=bert-base-cased`")
    
    # Format the dataset
    # If the model is not OPT, add the BOS token to the prompt
    if model.config.model_type != "opt":
        dataset_dic = {
                "prompt": [tokenizer.bos_token + instruction + tokenizer.sep_token for instruction in dataset["instruction"]],
                "chosen": [completion + tokenizer.eos_token for completion in dataset["chosen_response"]],
                "rejected": [completion + tokenizer.eos_token for completion in dataset["rejected_response"]],
            }
    else:
        dataset_dic = {
                "prompt": [instruction + tokenizer.sep_token for instruction in dataset["instruction"]],
                "chosen": [completion + tokenizer.eos_token for completion in dataset["chosen_response"]],
                "rejected": [completion + tokenizer.eos_token for completion in dataset["rejected_response"]],
            }
        
    dataset = Dataset.from_dict(dataset_dic)

    if training_args.do_eval:
        formatted_dataset = formatted_dataset.train_test_split(test_size=data_args.validation_split_percentage)

        logger.info(f"Train set size: {len(formatted_dataset['train']):,} | Validation set size: {len(formatted_dataset['test']):,}")
    
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
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        do_eval=training_args.do_eval,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size if training_args.do_eval else None,
        evaluation_strategy=training_args.evaluation_strategy if training_args.do_eval else None,
        eval_steps=training_args.eval_steps if training_args.do_eval else None,
        save_strategy=training_args.save_strategy,
        logging_strategy=training_args.logging_strategy,
        logging_steps=training_args.logging_steps,
        max_steps=training_args.max_steps,
        save_steps=training_args.save_steps,
        optim=training_args.optim,
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_steps=training_args.warmup_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        report_to=['wandb', 'codecarbon'] if extra_args.wandb_token is not None else ['codecarbon'],
        remove_unused_columns=False,
    )

    # Set up the DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset if not training_args.do_eval else formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"] if training_args.do_eval else None,
        beta=extra_args.beta,
        max_prompt_length=data_args.max_prompt_length,
        max_length=data_args.max_length,
    )

    # Train the model
    dpo_trainer.train()

    # Save the model
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.save_model(training_args.output_dir)
    dpo_trainer.model.save_pretrained(output_dir)

    logger.info("Training complete!")

    # Push the model checkpoint to the hub if needed
    if training_args.push_to_hub and training_args.hub_token is not None:

        logger.info("Pushing model to hub!")

        api = HfApi(
            token=training_args.hub_token,
        )

        future = api.upload_folder(
            repo_id=training_args.hub_model_id,
            folder_path=training_args.output_dir,
            run_as_future=True,
        )

        logger.info("Ouput directory being uploaded to the hub.")

        while not future.done():
            pass
        
        logger.info("Ouput directory uploaded to the hub!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune a language model on the Aira reward dataset via DPO.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python dpo-training.py --spec-file dpo-training-specs.yaml