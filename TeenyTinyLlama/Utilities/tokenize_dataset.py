import argparse
from datasets import load_dataset
from huggingface_hub import login
from accelerate import Accelerator
from transformers import AutoTokenizer

def main(args):

    if not args.token or not args.tokenizer_name or not args.dataset_name or not args.dataset_split:
        raise ValueError("One or more required arguments are missing")

    login(token=args.token)

    # initialize the accelerator
    accelerator = Accelerator()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Load the dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

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
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on every text in dataset",
        )
    
    # In case the dataset has the column `token_type_ids`, we will remove it
    if "token_type_ids" in dataset.column_names:
        dataset = dataset.remove_columns("token_type_ids")
    
    # Group texts together so that we have chunks of max_seq_length
    # We are not adding any special token here, like the eos_token,
    # because the tokenizers adds the bos_token by default
    def group_texts(examples):

        concatenated_examples = {
            k: [t for example in examples[k] for t in example] for k in examples.keys()
        }

        for k in concatenated_examples.keys():
            concatenated_examples[k] = concatenated_examples[k][:-1]
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size

        result = {
            k: [
                t[i : i + args.block_size]
                for i in range(0, total_length, args.block_size)
            ]
            for k, t in concatenated_examples.items()
        }

        return result

    with accelerator.main_process_first():
        dataset = dataset.map(
            group_texts,
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.block_size}",
        )
    
    # Add a column named `labels` wich is a copy of the `input_ids` column
    with accelerator.main_process_first():
        dataset = dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
            load_from_cache_file=True,
            desc="Adding labels to the dataset",
        )
    
    # split the dataset in train and validation sets
    dataset = dataset.train_test_split(test_size=args.test_size, shuffle=args.shuffle, seed=args.seed)
    print(dataset)

    # Push dataset to the hub
    dataset.push_to_hub(args.dataset_name + "-tokenized-" + str(args.block_size))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize a dataset")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset to tokenize")
    parser.add_argument("--dataset-split", type=str, help="Split of the dataset to tokenize")
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use")
    parser.add_argument("--block-size", type=int, help="Block size to use")
    parser.add_argument("--test-size", type=int, help="Test size to use")
    parser.add_argument("--shuffle", type=bool, help="Shuffle the dataset")
    parser.add_argument("--seed", type=int, help="Seed to use")
    parser.add_argument("--token", type=str, help="Hugging Face token")

    main(parser.parse_args())

# How to run:
# python tokenize_dataset.py --dataset-name nicholasKluge/portuguese-corpus-v3 --dataset-split train --tokenizer-name nicholasKluge/TeenyTinyLlama-460m --block-size 2048 --test-size 25000 --shuffle True --seed 42 --token <your_token>