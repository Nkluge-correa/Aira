import sys
import math
import logging
import argparse
import datasets
import transformers
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator)

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from codecarbon import OfflineEmissionsTracker

from datasets import load_dataset

# Set these environment variables for improved performance in modern Ampere GPUs (e.g., A100) 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    
    # Create the accelerator
    accelerator = Accelerator()

    # Set the logger
    logger = get_logger(args.logger_name)

    # Create configurations for the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity of other libraries
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # Load the model and the tokenizer
    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint_path, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint_path, revision=args.revision)
    
    # Load the evaluation dataset
    eval_dataset = load_dataset(
            'parquet',
            data_files={
                "test": f'{args.eval_folder_path}/test/*.parquet',
            },
            streaming=False)['test']

    # Set the format to `torch`
    eval_dataset = eval_dataset.with_format("torch") 
    
    # Create the Evaluation DataLoader
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=default_data_collator, 
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
    )
    
    # Prepare everything with `accelerator`.
    model, eval_dataloader = accelerator.prepare(
            model, eval_dataloader
        )
        
    # Create the emissions tracker
    tracker = OfflineEmissionsTracker(
        log_level="critical", # set to "critical" to silence codecarbon
        output_file=f"emissions.csv",
        tracking_mode='machine',
        country_iso_code='DEU', # set to your country's ISO code
    )

    logger.info(f"Running evaluation at step {args.completed_steps}.")

    model.eval()
    losses = []

    tracker.start()
    for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), position=0, leave=True, disable=not accelerator.is_local_main_process, unit=" samples",  desc="Validation")):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
    
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        eval_loss = torch.mean(losses)
        perplexity = float("inf")
    
    logger.info(f"Step {args.completed_steps} | Perplexity: {perplexity} | Evaluation Loss: {eval_loss} | Total Energy Consumption: {tracker._total_energy.kWh + args.total_energy_consumption}")
    
    # print the results as a markdown table
    print("| Step | Evaluation Loss | Perplexity | Total Energy Consumption |")
    print("| ---- | --------------- | ---------- |------------------------- |")
    print(f"| {args.completed_steps} | {eval_loss} | {perplexity} | {tracker._total_energy.kWh + args.total_energy_consumption} |")
    
    tracker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a checkpoint.")
    parser.add_argument("--logger_name", type=str, default="TeenyTinyLlama", help="Name of the logger.")
    parser.add_argument("--model_checkpoint_path", type=str, default="TeenyTinyLlama-460m", help="Path to the model checkpoint.")
    parser.add_argument("--revision", type=str, default="step100000", help="Revision of the model checkpoint.")
    parser.add_argument("--eval_folder_path", type=str, default="./data/eval", help="Path to the evaluation folder.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--completed_steps", type=int, default=100000, help="Number of steps completed.")
    parser.add_argument("--total_energy_consumption", type=float, default=9.149488284015176, help="Total energy consumption until the checkpoint.")
    args = parser.parse_args()
    main(args)

# How to run this script:
# python evaluation.py --logger_name "TeenyTinyLlama" --model_checkpoint_path "nicholasKluge/TeenyTinyLlama-460m" --revision "main" --eval_folder_path "/content/drive/MyDrive/portuguese-corpus-v3-tokenized-large/data" --per_device_eval_batch_size 2 --completed_steps 200000 --total_energy_consumption 18.5564449
