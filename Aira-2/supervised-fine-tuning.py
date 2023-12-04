import os
import sys
import math
import yaml
import torch
import wandb
import random
import logging
import argparse
from tqdm import tqdm
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, HfApi

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

def main(spec_file):

    # Load the arguments from the spec file
    with open(spec_file, "r") as stream:
        all_kwargs = yaml.safe_load(stream)

    # Get the arguments for the model, data, training, and extra arguments (wandb, accelerator, etc.) 
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

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(
        mixed_precision=extra_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        project_dir=training_args.output_dir)

    # Log the state of the accelerator
    logger.info(accelerator.state, main_process_only=False)

    # Set seed for reproducibility
    if training_args.seed is not None:
        set_seed(training_args.seed)

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
        
        dataset_df = dataset.to_pandas()

        column_names = dataset_df.columns.tolist()

        logger.info(f"Loaded dataset: {data_args.dataset_name} | Split: {data_args.dataset_split} | Number of examples: {len(dataset_df):,}")

    else:

        raise ValueError("No dataset provided. Try running with `dataset_name=nicholasKluge/instruct-aira-dataset`")

    if model_args.base_model is not None:

        # Load the tokenizer of the base model. We add here all the special tokens we want.
        tokenizer_kwargs = {
                "cache_dir": model_args.cache_dir,
                "use_fast": model_args.use_fast,
                "revision": model_args.model_revision,
                "use_auth_token": training_args.hub_token,
                "trust_remote_code": model_args.trust_remote_code,
                "bos_token": model_args.bos_token,
                "sep_token": model_args.sep_token,
                "pad_token": model_args.pad_token,
                "unk_token": model_args.unk_token,
                "eos_token": model_args.eos_token,
            }

        # Load the tokenizer of the base model
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_model, **tokenizer_kwargs)

        # Set the configuration for the `base_model`
        config_kwargs = {
                "cache_dir": model_args.cache_dir,
                "revision": model_args.model_revision,
                "use_auth_token": training_args.hub_token,
                "trust_remote_code": model_args.trust_remote_code,
                "output_hidden_states": model_args.output_hidden_states,
            }
        
        # Load the configuration of the base model
        configuration = AutoConfig.from_pretrained(model_args.base_model, **config_kwargs)

        # Load the pretrained model to be fine-tuned
        model = AutoModelForCausalLM.from_pretrained(
                model_args.base_model,
                config=configuration,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=training_args.hub_token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
        
        # Resize the token embeddings of the model to match the tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # Enable gradient checkpointing if `gradient_checkpointing=True`
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        
        logger.info(f"Model to train (base architecture): {model_args.base_model}")

    else:

        raise ValueError("No base model provided. Try running with `base_model=gpt2`")

    # if `model.config.model_type == "opt"` then we do not need to append the bos_token to the demonstrations,
    # since the OPT model already prepends the bos_token to the prompt.`
    if model.config.model_type != "opt":
        dataset_df['demonstrations'] = tokenizer.bos_token + dataset_df['prompt'] + tokenizer.sep_token + dataset_df['completion'] + tokenizer.eos_token

    else:
        dataset_df['demonstrations'] = dataset_df['prompt'] + tokenizer.sep_token + dataset_df['completion'] + tokenizer.eos_token 

    # Get the maximum length of the demonstrations if it is not provided
    if data_args.max_length is None:
        dataset_df['length'] = dataset_df['demonstrations'].apply(lambda x: len(tokenizer.encode(x)))
        data_args.max_length = dataset_df['length'].max()
        dataset_df.drop(columns=['length'], inplace=True)    
        logger.info(f"Maximum length of demonstrations set to `None`. Using max_length of demonstrations: {data_args.max_length}")

    # Turn the pandas dataframe into a HuggingFace Dataset
    dataset = Dataset.from_pandas(dataset_df)

    # Tokenize all texts in the dataset
    def tokenize_function(examples):
        return tokenizer(examples['demonstrations'],
            truncation=True,
            max_length=data_args.max_length,
            padding="max_length")

    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on every text in dataset",
        )

    # Add a column named `labels` wich is a copy of the `input_ids` column
    with accelerator.main_process_first():
        dataset = dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc="Adding labels to the dataset",
        )

    # Split the dataset into train and validation sets
    if training_args.do_eval and data_args.validation_split_percentage is not None:

        logger.info("Splitting the dataset into train and validation sets...")

        dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage)

        logger.info(f"Train set size: {len(dataset['train']):,} ({len(dataset['train']) * data_args.max_length:,} tokens)| Validation set size: {len(dataset['test']):,}")

    else:

        logger.info(f"Using the whole dataset for training. Training set size: {len(dataset):,}")

    # Create the Training DataLoader
    if training_args.do_train and training_args.do_eval:
        if "train" not in dataset:
            raise ValueError("`do_train=True` requires a train dataset")
        train_dataset = dataset["train"]
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

        # Create the Evaluation DataLoader
        if "test" not in dataset:
            raise ValueError("`do_eval=True` requires a validation dataset")
        eval_dataset = dataset["test"] 
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_eval_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    elif training_args.do_train and not training_args.do_eval:
        train_dataset = dataset
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # We set the `adam_epsilon` to 1e-5 if mixed precision is used. Otherwise we use the default value of 1e-8.
    # This helps avoid NANs as loss during mixed precision training.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon if extra_args.mixed_precision == "no" else 1e-5,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)

    # Create a scheduler to set the learning rate at each training step
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * training_args.num_train_epochs) * training_args.gradient_accumulation_steps,
    )

    # Prepare everything with `accelerator`.
    if training_args.do_train and training_args.do_eval:

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    else:

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Initialize W&B tracker if needed
    if extra_args.wandb_token is not None: 
        # Login to wandb    
        wandb.login(key=extra_args.wandb_token)

        # Initialize wandb
        wandb.init(
            project=extra_args.project_name, 
            notes="Fine tuning base model on the AIRA dataset",
            tags=["Alignment", "Fine-tuning", "Aira"],
            config=all_kwargs
        )

    # Intialize codecarbon tracker
    tracker = EmissionsTracker(
        project_name=extra_args.project_name,
        log_level="critical", # set to "critical" to silence codecarbon
        output_dir=training_args.output_dir,
        output_file=f"{extra_args.project_name}.csv",
        tracking_mode='machine'
    )

    logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')

    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {(len(train_dataloader) * training_args.num_train_epochs) * training_args.gradient_accumulation_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(train_dataloader) * training_args.num_train_epochs), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Start training loop and activate codecarbon tracking
    tracker.start()

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        logger.info(f'Beginning epoch {epoch + 1} of {training_args.num_train_epochs}')
        
        total_loss = 0
    
        # Iterate over the batches of data in the current epoch
        for step, batch in enumerate(train_dataloader, start=1):
            with accelerator.accumulate(model):

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()

                # Log the loss to wandb
                if (step) % extra_args.wandb_log_steps == 0 and extra_args.wandb_token is not None:
                    wandb.log({
                        "loss": loss.detach().float().item(),     
                        })

                # Backward pass and update optimizer
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update the progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            # Generate text from the model every `sample_every ` steps
            if step % extra_args.sample_every == 0 and not step == 0:
                
                model.config.use_cache = True

                try:

                    model.eval()

                    if model.config.model_type != "opt":
                        inputs = tokenizer(tokenizer.bos_token + dataset_df.prompt.sample().iloc[0] + tokenizer.sep_token, return_tensors="pt").to('cuda:0')
                        
                    else:
                        inputs = tokenizer(dataset_df.prompt.sample().iloc[0] + tokenizer.sep_token, return_tensors="pt").to('cuda:0')  

                    sample_outputs = model.generate(**inputs,
                                        bos_token_id=tokenizer.bos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        top_k=50,
                                        max_length=150,
                                        top_p=0.50,
                                        num_return_sequences=5)
                    
                    model.config.use_cache = False
                    
                    texts = []

                    for i, sample_output in enumerate(sample_outputs):
                        texts.append(tokenizer.decode(sample_output))
                    
                    for text in texts:
                        logger.info(f"Samples (Epoch: {epoch + 1} | Step: {step}): {text}")
                        
                    if extra_args.wandb_token is not None:

                        training_samples = wandb.Table(columns=[f"Samples (Epoch: {epoch + 1} | Step: {step})"])
                        for text in texts:
                            training_samples.add_data(text)
                        wandb.log({f"Samples (Epoch: {epoch + 1} | Step: {step})": training_samples})
                
                except Exception as e:
                    logger.warning(f"Error while generating samples: {e}")
                    model.config.use_cache = False

                model.train()
        
        # Evaluate the model at the end of each epoch if `do_eval=True`
        if training_args.do_eval:
            model.eval()
            losses = []
            logger.info(f"Running evaluation at the end of Epoch {epoch + 1}.")
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(batch["input_ids"], 
                                labels=batch["input_ids"], 
                                attention_mask=batch["attention_mask"])
                
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))
            
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                eval_loss = torch.mean(losses)
                perplexity = float("inf")
            
            logger.info(f"Epoch {epoch + 1} | Perplexity: {perplexity} | Average Training Loss: {total_loss.item() / completed_steps} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh}")
            
        
            # Log the metrics to wandb if needed
            if extra_args.wandb_token is not None:

                    wandb.log({
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,     
                        "avg_train_loss": total_loss.item() / completed_steps,
                        "total_energy_consumption": tracker._total_energy.kWh,      
                    })
            
        else:
            logger.info(f"Epoch {epoch + 1} | Average Training Loss: {total_loss.item() / completed_steps} | Total Energy Consumption: {tracker._total_energy.kWh}")

            # Log the metrics to wandb if needed
            if extra_args.wandb_token is not None:

                    wandb.log({    
                        "avg_train_loss": total_loss.item() / completed_steps,
                        "total_energy_consumption": tracker._total_energy.kWh,      
                    })
        
        # Save the model checkpoint at the end of each epoch    
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)   
        
    # Resume codecarbon tracking
    tracker.stop()
    logger.info("Training complete!")

    # Resume wandb tracking
    if extra_args.wandb_token is not None:
        wandb.finish()

    # Save the optimizer, lr_scheduler states, rng state, and pytorch model
    rng_state = torch.get_rng_state()
    torch.save(rng_state, f"./{training_args.output_dir}/rng_state.pt")
    torch.save(lr_scheduler.state_dict(), f"./{training_args.output_dir}/lr_scheduler.pt")
    torch.save(optimizer.state_dict(), f"./{training_args.output_dir}/optimizer.pt")

    # Push the model checkpoint to the hub if needed
    if training_args.push_to_hub and training_args.hub_token is not None:

        try:

            api = HfApi(
                token=training_args.hub_token,
            )

            logger.info(f"""Ouput directory (`{training_args.output_dir}`) being uploaded to the hub.""")

            api.upload_folder(
                repo_id=training_args.hub_model_id,
                folder_path=training_args.output_dir,
            )
            
            logger.info(f"Ouput directory (`{training_args.output_dir}`) uploaded to the hub!")
        
        except Exception as e:
            logger.warning(f"Error while uploading checkpoint to Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine tune a language model on the Aira dataset.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python supervised-fine-tuning.py --spec-file supervised-fine-specs.yaml