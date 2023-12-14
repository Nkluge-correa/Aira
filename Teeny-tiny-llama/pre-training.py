# For distributed training, run this script with the torch.distributed.launch utility.
# Example:
#
# python -m torch.distributed.launch --nproc_per_node 4 pre-training.py --spec-file specs.yaml
#
# or, if using `accelerate`:
#
# accelerate launch pre-training.py --spec-file specs.yaml
#
# This will launch 4 processes on the current node, each with 1 GPU device per process.
# More information can be found here: https://github.com/huggingface/transformers/tree/main/examples/pytorch#distributed-training-and-mixed-precision
import os
import sys
import time
import yaml
import math
import json
import random
import logging
import warnings
import argparse
from tqdm.auto import tqdm
from pathlib import Path

import torch
import wandb
import datasets
import transformers
import huggingface_hub
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
    GenerationConfig,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

# Set these environment variables for improved performance in the Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Make accelerator device be the CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    huggingface_hub.utils.logging.set_verbosity_error()

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
            "token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
        }

        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        tokenizer.push_to_hub(training_args.hub_model_id, token=training_args.hub_token)
    
    else:
        raise ValueError("Need a tokenizer name to train on. Train a tokenizer from scratch usign the `train_sentencepiece.py`.")

    # See if we need to train the model from scratch
    if model_args.train_from_scratch:

        logger.info("Training new model from scratch (train_from_stratch=True)")

        # Set the configuration kwargs to create a new model
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": training_args.hub_token,
            "trust_remote_code": model_args.trust_remote_code,
            "output_hidden_states": model_args.output_hidden_states,
            "hidden_size": model_args.hidden_size,
            "intermediate_size": model_args.intermediate_size,
            "max_position_embeddings": model_args.max_position_embeddings,
            "num_attention_heads": model_args.num_attention_heads,
            "num_hidden_layers": model_args.num_hidden_layers,
            "num_key_value_heads": model_args.num_key_value_heads,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "torch_dtype": "float16" if extra_args.mixed_precision.endswith('16') else model_args.torch_dtype,
            "vocab_size": len(tokenizer),
            "use_cache": model_args.use_cache,
        }
        
        # Load the configurations to create a new model
        configuration = AutoConfig.from_pretrained(model_args.model_to_train, **config_kwargs)
        model = AutoModelForCausalLM.from_config(configuration)
        model.config.name_or_path = training_args.hub_model_id

        # Resize the model's embedding layer to match the tokenizer's vocabulary size
        model.resize_token_embeddings(len(tokenizer))

        # Count the number of trainable parameters in the model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'There are {params:,} trainable parameters in this model.')

        # Create generation config file
        generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            max_length=model_args.max_position_embeddings,
        )

    else:

        logger.info("Fine-tuning model from HuggingFace Hub (train_from_stratch=False)")

        # Set the configuration kwargs for the model
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": training_args.hub_token,
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
                token=training_args.hub_token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
        
        model.config.name_or_path = training_args.hub_model_id
        # Resize the model's embedding layer to match the tokenizer's vocabulary size
        model.resize_token_embeddings(len(tokenizer))

        # Count the number of trainable parameters in the model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'There are {params:,} trainable parameters in this model.')

        # Load the generation config file
        generation_config = GenerationConfig.from_pretrained(model_args.model_to_train)

    # Set the gradient checkpointing if needed
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


    # Load the datasets
    #
    # Streaming datasets work faster if they are in a local directory
    # The dataset folder must contain a list o parquet files, and you 
    # can achieve this by simply cloning the dataset from the hub 
    # to a local directory:
    #
    # git lfs install
    # git clone https://huggingface.co/datasets/nicholasKluge/your-tokenized-dataset
    #
    # Then, you should separate the dataset into `train` and `test` folders.
    # To stream not locally, use `data_args.dataset_name` instead of `parquet`
    # and omit the `data_files`` argument.
    train_dataset = load_dataset(
            'parquet', 
            data_files={
                "train": './portuguese-corpus-v2-tokenized-2048/train/*.parquet',
            },  
            streaming=data_args.streaming,)['train']
    

    # We are not streaming the validation dataset, since it is small
    # and can be loaded into memory without any problems.
    eval_dataset = load_dataset(
            'parquet',
            data_files={
                "test": './portuguese-corpus-v2-tokenized-2048/test/*.parquet',
            },
            streaming=False,)['test']

    # Set the format to `torch`
    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch") 
    
    # Shuffle the `train_dataset` and set a buffer size if in streaming mode
    train_dataset = train_dataset.shuffle(seed=training_args.seed, buffer_size=data_args.buffer_size) \
        if data_args.streaming else train_dataset

    logger.info(f"Loaded dataset: {data_args.dataset_name} | Number of examples: {data_args.train_num_samples + data_args.val_num_samples:,}")
    logger.info(f"Size of train dataset: {data_args.train_num_samples:,} ({data_args.train_num_samples * data_args.block_size:,} tokens)| Size of validation dataset: {data_args.val_num_samples:,}")

    # If we wat to do a sanity check, we will use a small subset of the dataset
    if data_args.sanity_check:

        logger.info(f"`Sanity check` is set to `True`. Train set size: 400 | Validation set size: 40")

        train_dataset = train_dataset.take(400) if data_args.streaming else train_dataset.select(range(400)) 
        eval_dataset = eval_dataset.select(range(40))

        # Change the `total_num_samples` to reflect the size of the training set and the validation set
        data_args.train_num_samples, data_args.val_num_samples = 400, 40

    # Create the Training DataLoader and Evaluation DataLoade
    if training_args.do_train and training_args.do_eval:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=not data_args.streaming, # Streaming datasets do not support shuffling in the DataLoader
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_train_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=default_data_collator, 
            batch_size=training_args.per_device_eval_batch_size,
            pin_memory=training_args.dataloader_pin_memory,
        )
    
    # Create only the Training DataLoader
    elif training_args.do_train and not training_args.do_eval:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=not data_args.streaming, # Streaming datasets do not support shuffling in the DataLoader
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
        eps=training_args.adam_epsilon,
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(data_args.train_num_samples / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))

    # Create a scheduler to set the learning rate at each training step
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=(math.ceil(data_args.train_num_samples / training_args.per_device_train_batch_size) * training_args.num_train_epochs) * training_args.gradient_accumulation_steps,
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

    # Initialize the W&B tracker
    if extra_args.wandb_token is not None: 
        # Login to wandb    
        wandb.login(key=extra_args.wandb_token)

        # Initialize wandb
        wandb.init(
            project=extra_args.logger_name, 
            notes="Training the Teeny-Tiny-Llama model on a custom Portuguese-BR dataset.",
            tags=["Energy Consumption", "Language Modeling", "Portuguese"],
            #name=f"""{extra_args.logger_name.lower()}-{model_args.model_id}-{time.strftime("%d-%m-%Y")}""",
            config=all_kwargs,
            #resume="allow",
            #id=extra_args.logger_name.lower() + "-" + model_args.model_id,
        )

    # Intialize codecarbon tracker
    tracker = EmissionsTracker(
        project_name=extra_args.logger_name,
        log_level="critical", # set to "critical" to silence codecarbon
        output_dir=training_args.output_dir,
        output_file=f"emissions.csv",
        tracking_mode='machine'
    )

    logger.info(f'Geo Location: ISO: {tracker._geo.country_iso_code} | Country: {tracker._geo.country_name} | Region : {tracker._geo.region}')

    # Calculate the total batch size (important for distributed training)
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {data_args.train_num_samples + data_args.val_num_samples} | Training examples: {data_args.train_num_samples} | Validations examples: {data_args.val_num_samples}.")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {(math.ceil(data_args.train_num_samples / training_args.per_device_train_batch_size) * training_args.num_train_epochs) * training_args.gradient_accumulation_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(math.ceil(data_args.train_num_samples / training_args.per_device_train_batch_size) * training_args.num_train_epochs), disable=not accelerator.is_local_main_process, unit=" samples", desc="Training")
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
            starting_epoch = resume_step // math.ceil(data_args.train_num_samples/training_args.per_device_train_batch_size)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * math.ceil(data_args.train_num_samples/training_args.per_device_train_batch_size)
    
    # Update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    accelerator.print()

    # Start training loop and activate codecarbon tracking
    tracker.start()

    for epoch in range(starting_epoch, training_args.num_train_epochs):

        # Since our dataset is a streaming dataset, we need to reset the epoch at each iteration
        model.train()
        train_dataset.set_epoch(epoch) if data_args.streaming else None
        logger.info(f'Beginning epoch {epoch + 1} of {training_args.num_train_epochs}')

        total_loss = 0

        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            logger.info(f"Skipping the first {resume_step} batches of the first epoch.")
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
            if isinstance(extra_args.checkpointing_steps, int):
                if completed_steps % extra_args.checkpointing_steps == 0 and completed_steps > 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # Save the model checkpoint
                    accelerator.save_state(output_dir)
                    # Save the generation config file
                    generation_config.save_pretrained(output_dir)

                    # Flush the codecarbon tracker
                    tracker.flush()
                
                    # Push the model checkpoint to the hub if needed
                    if training_args.push_to_hub and training_args.hub_token is not None:
                        if training_args.hub_model_id is not None:

                            accelerator.wait_for_everyone()

                            # Handle the repository creation if needed
                            create_repo(
                                repo_id=f"{training_args.hub_model_id}-step-{completed_steps}",  
                                token=training_args.hub_token,
                                repo_type="model",
                                exist_ok=True,
                                private=True)
        
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                            )

                            if accelerator.is_main_process:

                                try:

                                    logger.info(f"""Checkpoint directory (`{output_dir}`) being uploaded to the hub.""")

                                    api = HfApi(
                                        token=training_args.hub_token,
                                    )

                                    api.upload_folder(
                                        repo_id=f"{training_args.hub_model_id}-step-{completed_steps}", 
                                        folder_path=output_dir,
                                    )

                                    api.upload_file(
                                        path_or_fileobj=f"./{training_args.output_dir}/emissions.csv",
                                        path_in_repo=f"emissions.csv",
                                        repo_id=f"{training_args.hub_model_id}-step-{completed_steps}", 
                                    )

                                    tokenizer.push_to_hub(f"{training_args.hub_model_id}-step-{completed_steps}", token=training_args.hub_token)

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

                        accelerator.print()
                        logger.info(f"Running evaluation at step {completed_steps}.")

                        model.eval()
                        losses = []
                        for step, batch in enumerate(tqdm(eval_dataloader, total=data_args.val_num_samples / training_args.per_device_eval_batch_size, position=0, leave=True, disable=not accelerator.is_local_main_process, unit=" samples",  desc="Validation")):
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
            accelerator.print()
            logger.info(f"Running evaluation at the end of Epoch {epoch + 1}.")

            for step, batch in enumerate(tqdm(eval_dataloader, total=data_args.val_num_samples / training_args.per_device_eval_batch_size, position=0, leave=True, disable=not accelerator.is_local_main_process, unit=" samples",  desc="Validation")):
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
        # Save the generation config file
        generation_config.save_pretrained(output_dir)

        # Flush the codecarbon tracker
        tracker.flush()

        # Push the model checkpoint to the hub if needed
        if training_args.push_to_hub and training_args.hub_token is not None: 
            if training_args.hub_model_id is not None:
                
                accelerator.wait_for_everyone()

                # Handle the repository creation if needed
                create_repo(
                    repo_id=f"{training_args.hub_model_id}-step-{completed_steps}", 
                    token=training_args.hub_token,
                    repo_type="model",
                    exist_ok=True,
                    private=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )

                if accelerator.is_main_process:

                    try:

                        logger.info(f"""Checkpoint directory (`{output_dir}`) being uploaded to the hub.""")

                        api = HfApi(
                            token=training_args.hub_token,
                        )

                        api.upload_folder(
                            repo_id=f"{training_args.hub_model_id}-step-{completed_steps}",  
                            folder_path=output_dir,
                        )

                        api.upload_file(
                            path_or_fileobj=f"./{training_args.output_dir}/emissions.csv",
                            path_in_repo=f"emissions.csv",
                            repo_id=f"{training_args.hub_model_id}-step-{completed_steps}", 
                        )
                        
                        tokenizer.push_to_hub(f"{training_args.hub_model_id}-step-{completed_steps}", token=training_args.hub_token)

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

        # Save the generation config file
        generation_config.save_pretrained(training_args.output_dir)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub and training_args.hub_token is not None:
        if training_args.hub_model_id is not None:

            try:
                
                unwrapped_model.push_to_hub(
                        repo_id=training_args.hub_model_id,
                        commit_message=f"Training complete!",
                    )
                
                api.upload_file(
                    path_or_fileobj=f"./{training_args.output_dir}/emissions.csv",
                    path_in_repo=f"emissions.csv",
                    repo_id=training_args.hub_model_id,
                )

                generation_config.push_to_hub(
                        repo_id=training_args.hub_model_id,
                        commit_message=f"Training complete!",
                        token=training_args.hub_token,
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