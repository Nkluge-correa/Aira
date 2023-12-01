import yaml
import argparse
from tqdm import tqdm

import torch
import datasets
from datasets import load_dataset 

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    GPT2TokenizerFast,
)

from specifications import ModelArguments, DataTrainingArguments, ExtraArguments

def main(spec_file):
    
    # Load the arguments from the spec file
    with open(spec_file, "r") as stream:
        kwargs = yaml.safe_load(stream)
    
    # Get the arguments for the model, data, training, and extra
    model_args = ModelArguments(**kwargs['model_args'])
    data_args = DataTrainingArguments(**kwargs['data_args'])
    training_args = TrainingArguments(**kwargs['training_args'])
    extra_args = ExtraArguments(**kwargs['extra_args'])

    # Load the dataset from the huggingface Hub and prepare it for training
    if data_args.dataset_name is not None and not data_args.dataset_is_tokenized:
        dataset = load_dataset(data_args.dataset_name, 
            split=data_args.dataset_split, 
            use_auth_token=training_args.hub_token if training_args.hub_token else None,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
    else:
        raise ValueError("No dataset name provided or dataset is already tokenized") 

    # Remove non text columns
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    
    # create a python generator to dynamically load the data
    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i : i + batch_size]["text"]

    # Create a tokenizer from the model checkpoint you want to train
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name, 
        use_auth_token=training_args.hub_token if training_args.hub_token is not None else None
    )

    new_tokenizer = tokenizer.train_new_from_iterator(
        text_iterator=batch_iterator(), 
        vocab_size=32000,
    )

    # Replace the new_tokenizer `max_model_input_sizes` for the `data_args.block_size`
    new_tokenizer.max_model_input_sizes.clear()
    new_tokenizer.max_model_input_sizes[extra_args.logger_name] = data_args.block_size
    new_tokenizer.model_max_length = data_args.block_size

    # Add the pad_token as the eos_token
    new_tokenizer.pad_token = new_tokenizer.eos_token

    # Save the new tokenizer
    new_tokenizer.save_pretrained(training_args.output_dir)
    
    # Wrap the new tokenizer in a fast tokenizer
    new_tokenizer_fast = GPT2TokenizerFast(
        tokenizer_file=f'{training_args.output_dir}/tokenizer.json',
    )

    # Add the special tokens to the fast tokenizer
    new_tokenizer_fast.bos_token=new_tokenizer.bos_token
    new_tokenizer_fast.bos_token_id=new_tokenizer.convert_tokens_to_ids(new_tokenizer.bos_token)
    new_tokenizer_fast.eos_token=new_tokenizer.eos_token
    new_tokenizer_fast.eos_token_id=new_tokenizer.convert_tokens_to_ids(new_tokenizer.eos_token)
    new_tokenizer_fast.unk_token=new_tokenizer.unk_token
    new_tokenizer_fast.unk_token_id=new_tokenizer.convert_tokens_to_ids(new_tokenizer.unk_token)
    new_tokenizer_fast.pad_token=new_tokenizer.pad_token
    new_tokenizer_fast.pad_token_id=new_tokenizer.convert_tokens_to_ids(new_tokenizer.pad_token)

    # Set the fast tokenizer max model input sizes
    new_tokenizer_fast.max_model_input_sizes = new_tokenizer.max_model_input_sizes
    
    # Save the new tokenizers
    new_tokenizer.save_pretrained(save_directory=training_args.output_dir)
    new_tokenizer_fast.save_pretrained(save_directory=training_args.output_dir)

    # If hub_token is passed, upload the tokenizer to the hub
    if training_args.hub_token is not None and training_args.hub_model_id is not None:
        new_tokenizer.push_to_hub(
            repo_id=training_args.hub_model_id,
            use_auth_token=training_args.hub_token,
            commit_message=f"Trained tokenizer from scratch on {data_args.dataset_name}",
        )

        new_tokenizer_fast.push_to_hub(
            repo_id=training_args.hub_model_id,
            use_auth_token=training_args.hub_token,
            commit_message=f"GPT2TokenizerFast implementation",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer from scratch")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python train_tokenizer.py --spec-file specs.yaml