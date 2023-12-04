# For distributed training, run this script with the torch.distributed.launch utility.
# Example:
#
# python -m torch.distributed.launch --nproc_per_node 4 pre-training.py --spec-file specs.yaml
#
# This will launch 4 processes on the current node, each with 1 GPU device per process.
# More information can be found here: https://github.com/huggingface/transformers/tree/main/examples/pytorch#distributed-training-and-mixed-precision
import os
import sys
import yaml
import math
import json
import random
import logging
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import wandb
import datasets
import transformers
from datasets import load_dataset
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
from huggingface_hub import create_repo, HfApi

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

def main(spec_file):

    # Load the arguments from the spec file
    with open(spec_file, "r") as stream:
        all_kwargs = yaml.safe_load(stream)

    # Get the arguments for the model, data, training, and extra arguments (wandb, accelerate, etc.)
    model_args = ModelArguments(**all_kwargs['model_args'])
    data_args = DataTrainingArguments(**all_kwargs['data_args'])
    training_args = TrainingArguments(**all_kwargs['training_args'])
    extra_args = ExtraArguments(**all_kwargs['extra_args'])

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision=extra_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps, 
        project_dir=training_args.output_dir)

    # Set the logger
    logger = get_logger(extra_args.logger_name)

    # Create configurations for the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    # Set seed before initializing model.
    if training_args.seed is not None:
        set_seed(training_args.seed)
    
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

    accelerator.wait_for_everyone()

    # Load the portuguese tokenizer
    if model_args.tokenizer_name is not None:

        # Set the configuration kwargs for the tokenizer
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "bos_token": model_args.bos_token,
            "sep_token": model_args.sep_token,
            "pad_token": model_args.pad_token,
            "unk_token": model_args.unk_token,
            "eos_token": model_args.eos_token,
        }

        # Clear the tokenizer's `max_model_input_sizes` 
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        tokenizer.max_model_input_sizes.clear()
    
    else:
        raise ValueError("Need a tokenizer name to train on. Train a tokenizer from scratch usign the `train_tokenizer.py`.")

    # See if we need to train the model from scratch
    if model_args.train_from_scratch:

        logger.info("Training new model from scratch (train_from_stratch=True)")

        # Set the configuration kwargs to create a new model
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
            "hidden_size": model_args.hidden_size,
            "intermediate_size": model_args.intermediate_size,
            "max_position_embeddings": model_args.max_position_embeddings,
            "num_attention_heads": model_args.num_attention_heads,
            "num_hidden_layers": model_args.num_hidden_layers,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "vocab_size": len(tokenizer),
        }
        
        # Load the configurations to create a new model
        configuration = AutoConfig.from_pretrained(model_args.model_to_train, **config_kwargs)
        model = AutoModelForCausalLM.from_config(configuration)

        # Count the number of trainable parameters in the model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'There are {params:,} trainable parameters in this model.')

    else:

        # Set the configuration kwargs for the model
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
        }

        # Load the configuration of the model to train
        configuration = AutoConfig.from_pretrained(model_args.model_to_train, **config_kwargs)

        # Load the pretrained model to fine-tune
        model = AutoModelForCausalLM.from_pretrained(
                model_args.model_to_train,
                config=configuration,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=training_args.hub_token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )

    # Replace the tokenizer `max_model_input_sizes` to the maximum sequence length of the new model
    tokenizer.max_model_input_sizes[extra_args.logger_name] = model.config.max_position_embeddings

    # if `block_size` is None, we will use the model's own max length
    if data_args.block_size is None:
        data_args.block_size = model.config.max_position_embeddings
    
    # if `block_size` is larger than the model's own max length, we will use the model's own max length
    elif data_args.block_size > model.config.max_position_embeddings:
        data_args.block_size = model.config.max_position_embeddings
        logger.info(f"Block size set to the model's own max length: {data_args.block_size}")

    # Resize the model's embedding layer to match the tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer))

    # Clone the embedding layer from another model
    # If `clone_embedding_layer=True` this script will clone the embedding layer of the specified 
    # `embedding_model`, e.g., "pierreguillou/gpt2-small-portuguese" has embeddings trained in protuguese
    if model_args.clone_embedding_layer:

        logger.info(f"Model to train (base architecture): {model_args.model_to_train} | Model to clone embeddings: {model_args.embedding_model}")

        # Set the configuration kwargs for the model
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
        }

        # Load the model to clone embeddings
        embedding_model_configuration = AutoConfig.from_pretrained(model_args.embedding_model, **config_kwargs)
        
        model_embedding = AutoModelForCausalLM.from_pretrained(
                model_args.embedding_model,
                config=embedding_model_configuration,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=training_args.hub_token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
        
        # Resize the model's embedding layer to match the tokenizer's vocabulary size
        model_embedding.resize_token_embeddings(len(tokenizer))

        # Clone the embedding layer from the model to clone embeddings if the dimensions match
        if model.get_input_embeddings().weight.shape[1] == model_embedding.get_input_embeddings().weight.shape[1]:
            model.get_input_embeddings().weight.data = model_embedding.get_input_embeddings().weight.data
            logger.info(f"Embeddings/tokenizer cloned from {model_args.embedding_model} to {model_args.model_to_train}")
        
        else:
            # If dimensions don't match, pad the embedding layer of the model
            # Get the dimensions of the original and target embedding layers
            original_dim = model_embedding.get_input_embeddings().weight.shape[1]
            target_dim = model.get_input_embeddings().weight.shape[1] 

            # Pad the embedding layer of the model to match the target dimensions
            if original_dim < target_dim:
                # If the target dimensions are larger, pad the original embedding layer
                pad_dim = target_dim - original_dim
                padding = torch.zeros((model.get_input_embeddings().weight.shape[0], pad_dim))
                new_weight = torch.cat([model_embedding.get_input_embeddings().weight.data, padding], dim=1)
                model.get_input_embeddings().weight.data = new_weight

                logger.info(f"Embeddings/tokenizer cloned from {model_args.embedding_model}. Embedding layer padded from {original_dim} to {target_dim}")

            elif original_dim > target_dim:
                # If the target dimensions are smaller, truncate the original embedding layer
                new_weight = model_embedding.get_input_embeddings().weight.data[:, :target_dim]
                model.get_input_embeddings().weight.data = new_weight

                logger.info(f"Embeddings/tokenizer cloned from {model_args.embedding_model}. Embedding layer truncated from {original_dim} to {target_dim}")
    else:
        logger.info(f"Model to train (base architecture): {model_args.model_to_train} | (clone_embedding_layer=False)")

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load the dataset from the huggingface Hub and prepare it for training
    if data_args.dataset_name is not None:
        dataset = load_dataset(data_args.dataset_name, 
            split=data_args.dataset_split, 
            use_auth_token=training_args.hub_token if training_args.hub_token else None,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )       

        logger.info(f"Loaded dataset: {data_args.dataset_name} | Split: {data_args.dataset_split} | Number of examples: {len(dataset):,}")

        # Sanity check: use only the first 100 examples
        if data_args.sanity_check:
            dataset = dataset.select(range(100))

            logger.info(f"Sanity check: using only the first 100 examples")

    else:
        raise ValueError("Need a dataset name to train on.")

    # Preprocessing the dataset
    if data_args.dataset_is_tokenized:

        # Load the dataset as torch tensors
        dataset = dataset.with_format("torch")
        logger.info(f"Dataset `{data_args.dataset_name}` is already tokenized. Using it as is...")
    
    else:

        # Get the column names
        column_names = dataset.column_names

        # Get the text column name
        text_column_name = "text" if "text" in column_names else column_names[0]
        
        # Tokenize all texts in the dataset
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])
        
        with accelerator.main_process_first():
            dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on every text in dataset",
            )
        
        # Group texts together so that we have chunks of max_seq_length
        def group_texts(examples):
            eos_token_id = tokenizer.eos_token_id

            concatenated_examples = {
                k: [t for example in examples[k] for t in example + [eos_token_id]] for k in examples.keys()
            }

            for k in concatenated_examples.keys():
                concatenated_examples[k] = concatenated_examples[k][:-1]
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            if total_length >= data_args.block_size:
                total_length = (total_length // data_args.block_size) * data_args.block_size

            result = {
                k: [
                    t[i : i + data_args.block_size]
                    for i in range(0, total_length, data_args.block_size)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        with accelerator.main_process_first():
            dataset = dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                desc=f"Grouping texts in chunks of {data_args.block_size}",
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

        logger.info(f"Train set size: {len(dataset['train']):,} ({len(dataset['train']) * data_args.block_size:,} tokens)| Validation set size: {len(dataset['test']):,}")
    
    else:

        logger.info(f"Using the whole dataset for training. Training set size: {len(dataset):,}")

    # Create the Training DataLoader and Evaluation DataLoade
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

        if "test" not in dataset:
            raise ValueError("`do_eval=True` requires a validation dataset")
        eval_dataset = dataset["test"] 
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_eval_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    # Create only the Training DataLoader
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
    
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = extra_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Initialize the W&B tracker
    if extra_args.wandb_token is not None: 
        # Login to wandb    
        wandb.login(key=extra_args.wandb_token)

        # Initialize wandb
        wandb.init(
            project=extra_args.logger_name, 
            notes="Training the Teeny-Tiny Llama model on a custom Portuguese-BR dataset.",
            tags=["Energy Consumption", "Language Modeling", "Portuguese"],
            config=all_kwargs
        )

    # Intialize codecarbon tracker
    tracker = EmissionsTracker(
        project_name=extra_args.logger_name,
        log_level="critical", # set to "critical" to silence codecarbon
        output_dir=training_args.output_dir,
        output_file=f"{extra_args.logger_name}.csv",
        tracking_mode='machine'
    )

    logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')

    # Calculate the total batch size (important for distributed training)
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

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    
    # Update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Start training loop and activate codecarbon tracking
    tracker.start()

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        logger.info(f'Beginning epoch {epoch + 1} of {training_args.num_train_epochs}')

        total_loss = 0

        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        # Iterate over the batches of data in the current epoch
        for step, batch in enumerate(active_dataloader, start=1):
            with accelerator.accumulate(model):
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
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
            
            # Save the model checkpoint if needed
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                
                    # Push the model checkpoint to the hub if needed
                    if training_args.push_to_hub and training_args.hub_token is not None:
                        if training_args.hub_model_id is not None:

                            accelerator.wait_for_everyone()

                            # Handle the repository creation if needed
                            create_repo(
                                repo_id=training_args.hub_model_id + f"-step-{completed_steps}", 
                                token=training_args.hub_token,
                                repo_type="model",
                                exist_ok=True,
                                private=True)
        
                            #unwrapped_model = accelerator.unwrap_model(model)
                            #unwrapped_model.save_pretrained(
                            #    training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                            #)

                            if accelerator.is_main_process:

                                try:

                                    logger.info(f"""Checkpoint directory (`{output_dir}`) being uploaded to the hub.""")

                                    api = HfApi(
                                        token=training_args.hub_token,
                                    )

                                    api.upload_folder(
                                        repo_id=training_args.hub_model_id + f"-step-{completed_steps}", 
                                        folder_path=output_dir,
                                    )

                                    logger.info(f"Checkpoint pushed to the hub at step {completed_steps}!")
                                
                                except Exception as e:
                                    logger.warning(f"Error while uploading checkpoint to Hub: {e}")
                                
            # Generate text from the model every `sample_every ` steps
            if step % extra_args.sample_every == 0 and not step == 0:
                
                model.config.use_cache = True

                try:

                    model.eval()

                    inputs = tokenizer(random.choice(extra_args.generation_seeds), return_tensors="pt").to('cuda:0')

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
            
            # Check if evaluation is needed
            if training_args.do_eval:
                # Check if `evaluation_strategy=steps`
                if training_args.evaluation_strategy == "steps":

                    if step % training_args.eval_steps == 0 and step > 0:

                        logger.info(f"Running evaluation at step {completed_steps}.")

                        model.eval()
                        losses = []
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                
                                outputs = model(**batch)
                            
                            loss = outputs.loss
                            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))
                        
                        losses = torch.cat(losses)
                        try:
                            eval_loss = torch.mean(losses)
                            perplexity = math.exp(eval_loss)
                        except OverflowError:
                            eval_loss = torch.mean(losses)
                            perplexity = float("inf")
                        
                        logger.info(f"Step {completed_steps} | Perplexity: {perplexity} | Average Training Loss: {total_loss.item() / completed_steps} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh}")
                        

                        accelerator.log(
                            {
                                "perplexity": perplexity,
                                "eval_loss": eval_loss,
                                "avg_train_loss": total_loss.item() / completed_steps,
                                "epoch": epoch + 1,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                        
                        # Log the metrics to wandb if needed
                        if extra_args.wandb_token is not None:

                                wandb.log({
                                    "eval_loss": eval_loss,
                                    "perplexity": perplexity,     
                                    "avg_train_loss": total_loss.item() / completed_steps,
                                    "total_energy_consumption": tracker._total_energy.kWh,      
                                })
                                
                                wandb.alert(title="Validation complete!",
                                    text=f"Current trainin stats -- Epoch: {epoch + 1} | Completed Steps: {completed_steps} | Evaluation Loss: {eval_loss} | Perplexity: {perplexity} | Total Energy Consumption: {tracker._total_energy.kWh}", 
                                    level="INFO")

        # Check if evaluation is needed in the end of the epoch
        if training_args.do_eval:
            # Evaluate the model at the end of each epoch
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
            
            logger.info(f"Epoch {epoch + 1} | Step {completed_steps} | Perplexity: {perplexity} | Average Training Loss: {total_loss.item() / completed_steps} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh}")
            
            # Log the metrics to wandb if needed
            if extra_args.wandb_token is not None:

                    wandb.log({
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,     
                        "avg_train_loss": total_loss.item() / completed_steps,
                        "total_energy_consumption": tracker._total_energy.kWh,      
                    })
                    
                    wandb.alert(title="Epoch complete!",
                        text=f"Current trainin stats -- Epoch: {epoch + 1} | Completed Steps: {completed_steps} | Evaluation Loss: {eval_loss} | Perplexity: {perplexity} | Total Energy Consumption: {tracker._total_energy.kWh}", 
                        level="INFO")
        
        else:

            logger.info(f"Epoch {epoch + 1} | Step {completed_steps} | Average Training Loss: {total_loss.item() / completed_steps} | Total Energy Consumption: {tracker._total_energy.kWh}")

            # Log the metrics to wandb if needed
            if extra_args.wandb_token is not None:

                    wandb.log({   
                        "avg_train_loss": total_loss.item() / completed_steps,
                        "total_energy_consumption": tracker._total_energy.kWh,      
                    })
                    
                    wandb.alert(title="Epoch complete!",
                        text=f"Current trainin stats -- Epoch: {epoch + 1} | Completed Steps: {completed_steps} | Total Energy Consumption: {tracker._total_energy.kWh}", 
                        level="INFO")

        # Save the model checkpoint at the end of each epoch
        output_dir = f"epoch_{epoch + 1}"
        if training_args.output_dir is not None:
            output_dir = os.path.join(training_args.output_dir, output_dir)
        accelerator.save_state(output_dir)

        # Push the model checkpoint to the hub if needed
        if training_args.push_to_hub and training_args.hub_token is not None: 
            if training_args.hub_model_id is not None:
                
                accelerator.wait_for_everyone()

                # Handle the repository creation if needed
                create_repo(
                    repo_id=training_args.hub_model_id + f"-step-{completed_steps}", 
                    token=training_args.hub_token,
                    repo_type="model",
                    exist_ok=True,
                    private=True)
                
                #unwrapped_model = accelerator.unwrap_model(model)
                #unwrapped_model.save_pretrained(
                #    training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                #)

                if accelerator.is_main_process:

                    try:

                        logger.info(f"""Checkpoint directory (`{output_dir}`) being uploaded to the hub.""")

                        api = HfApi(
                            token=training_args.hub_token,
                        )

                        api.upload_folder(
                            repo_id=training_args.hub_model_id + f"-step-{completed_steps}", 
                            folder_path=output_dir,
                        )
                        
                        logger.info(f"Checkpoint pushed to the hub at the end of epoch {epoch + 1}. Completed steps: {completed_steps}.")

                    except Exception as e:
                        logger.warning(f"Error while uploading checkpoint to Hub: {e}")

    # Resume codecarbon tracking
    logger.info("Training complete!")
    tracker.stop()

    # Resume wandb tracking
    if extra_args.wandb_token is not None:
        wandb.alert(title="Training complete!", text="Training complete!", level="INFO")
        wandb.finish()

    # Resume the tracking of the accelerator if needed
    if extra_args.with_tracking:
        accelerator.end_training()
    
    # Save the model checkpoint at the end of training, push it to the hub if needed
    if training_args.output_dir is not None:

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is not None:

            try:
                
                unwrap_model.push_to_hub(
                        repo_id=training_args.hub_model_id,
                        commit_message=f"Training complete!",
                    )
                
                api.upload_file(
                    path_or_fileobj=f"./{training_args.output_dir}/emissions.csv",
                    path_in_repo=f"emissions.csv",
                    repo_id=training_args.hub_model_id,
                )

                logger.info(f"Final model and emissions pushed to the hub!")
                            
            except Exception as e:
                logger.warning(f"Error while uploading checkpoint to Hub: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Teeny-tiny-llama on a \
        custom Portuguese-BR dataset.")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python pre-training.py --spec-file specs.yaml