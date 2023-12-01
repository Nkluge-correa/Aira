import evaluate
import itertools
from datasets.dataset_dict import DatasetDict
from datasets import concatenate_datasets, load_dataset

def get_data(list_of_data_args, data_args):

    _data_sets = []

    for data_arg in list_of_data_args:
        _data = load_dataset(
                data_arg.dataset_name,
                data_arg.dataset_config_name,
            )

        _data = _data['train'].train_test_split(test_size=data_args.test_split_percentage)
        
        _data_sets.append(_data)

        data = DatasetDict()
        for _type in ["train", "test"]:  
                data[_type] = concatenate_datasets([_data[_type] for _data in _data_sets])

    return data


def preprocess_data(data, data_args, training_args, tokenizer, logger):

    if training_args.do_train:
        column_names = data["train"].column_names

    else:
        column_names = data["test"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.info(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024

    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.info(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger "
                f"than the maximum length for the model ({tokenizer.model_max_length})."
                f" Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
                examples[text_column_name] = [
                    line
                    for line in examples[text_column_name]
                    if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = data.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

    else:
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = data.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )
        
        def group_texts(examples):
            concatenated_examples = {
                k: list(itertools.chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = None

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    else:
        train_dataset = concatenate_datasets([train_dataset, tokenized_datasets["test"]])
        eval_dataset = None
        compute_metrics = None
        preprocess_logits_for_metrics = None

    return train_dataset, eval_dataset, compute_metrics, preprocess_logits_for_metrics