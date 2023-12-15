import json
import yaml
import argparse
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer

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
            token=training_args.hub_token if training_args.hub_token else None,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
        )
    else:
        raise ValueError("No dataset name provided or dataset is already tokenized") 

    # Remove non text columns
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    # select 2_000_000 random samples from the dataset
    dataset = dataset.shuffle(seed=training_args.seed).select(range(2_000_000))

    # Create a SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer()

    # Train the SentencePieceBPETokenizer on the dataset
    tokenizer.train_from_iterator(
        iterator=dataset['text'],
        vocab_size=model_args.vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>",  "<pad>"],
    )

    # Save the tokenizer
    tokenizer.save(extra_args.logger_name + "-sentencepiece-tokenizer.json", pretty=True)

    # Load reference tokenizer
    if model_args.tokenizer_name is not None and training_args.hub_token is not None:
        reference_tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, token=training_args.hub_token if training_args.hub_token else None)
        reference_tokenizer.save_pretrained("reference-tokenizer")
    else:
        raise ValueError("No tokenizer name provided or no hub token provided. Try using `tokenizer_name=meta-llama/Llama-2-7b`")

    # Read and dump the json file for the new tokenizer and the reference tokenizer
    with open(extra_args.logger_name + "-sentencepiece-tokenizer.json") as f:
        new_llama_tokenizer_json = json.load(f)

    with open("reference-tokenizer/tokenizer.json") as f:
        reference_tokenizer_json = json.load(f)
    
    # Add the reference tokenizer's config to the new tokenizer's config
    new_llama_tokenizer_json["normalizer"] = reference_tokenizer_json["normalizer"]
    new_llama_tokenizer_json["pre_tokenizer"] = reference_tokenizer_json["pre_tokenizer"]
    new_llama_tokenizer_json["post_processor"] = reference_tokenizer_json["post_processor"]
    new_llama_tokenizer_json["decoder"] = reference_tokenizer_json["decoder"]
    new_llama_tokenizer_json["model"]['fuse_unk'] = reference_tokenizer_json["model"]['fuse_unk']
    new_llama_tokenizer_json["model"]['byte_fallback'] = reference_tokenizer_json["model"]['byte_fallback']

    # Dump the new tokenizer's config
    with open(extra_args.logger_name + "-sentencepiece-tokenizer.json", "w") as f:
        json.dump(new_llama_tokenizer_json, f, indent=2, ensure_ascii=False)

    # Load the new tokenizer as a LlamaTokenizerFast
    new_llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file=extra_args.logger_name + "-sentencepiece-tokenizer.json",
        name_or_path=training_args.hub_model_id + "-tokenizer",
        unk_token="<unk>",
        unk_token_id=0,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=3,
        padding_side="right",
        max_model_input_sizes={extra_args.logger_name: data_args.block_size},
    )

    # Save the new tokenizer
    new_llama_tokenizer.save_pretrained(extra_args.logger_name + "-tokenizer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new Llama tokenizer")
    parser.add_argument("--spec-file", help="Path to the spec YAML file")
    args = parser.parse_args()
    main(args.spec_file)

# How to run:
# python train_sentencepiece.py --spec-file specs.yaml
