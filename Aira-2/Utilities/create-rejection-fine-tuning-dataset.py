import tqdm
import torch
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

def main(args):
    # download the dataset
    dataset = load_dataset(args.dataset_name, split=args.split)

    # Select only the prompt column and
    prompts = dataset['prompt']

    # Load the models (chatbot, reward model)
    model = AutoModelForCausalLM.from_pretrained(args.generation_model)
    tokenizer = AutoTokenizer.from_pretrained(args.generation_model)

    rewardModel = AutoModelForSequenceClassification.from_pretrained(args.reward_model)
    rewardTokenizer = AutoTokenizer.from_pretrained(args.reward_model)

    # Specify the device (cuda if available) and move the models to the device
    device = torch.device(args.device_name)
    model.to(device)
    rewardModel.to(device)

    # Format the prompts
    if model.config.model_type != "opt":
        formatedPrompts = [tokenizer.bos_token + x + tokenizer.sep_token for x in prompts]
    else:
        formatedPrompts = [x + tokenizer.sep_token for x in prompts]

    # Generate the new responses by ordering the generations by reward
    new_responses = []

    for prompt in tqdm.tqdm(formatedPrompts):
        # Generate the responses
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_responses = model.generate(**inputs,
                    bos_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    early_stopping=True,
                    renormalize_logits=True,
                    top_k=args.top_k,
                    max_length=args.max_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_return_sequences=args.num_return_sequences)
        
        # Decode the responses
        texts = []
        for response in generated_responses:
            texts.append(tokenizer.decode(response, skip_special_tokens=True).replace(prompt, ""))

        # Get the reward for each response
        rewards = []
        for text in texts:
                reward_tokens = rewardTokenizer(prompt, text,
                            truncation=True,
                            max_length=512,
                            return_token_type_ids=False,
                            return_tensors="pt",
                            return_attention_mask=True)
                
                reward_tokens.to(rewardModel.device)
                
                reward = rewardModel(**reward_tokens)[0].item()
                rewards.append(reward)
        
        # Order the generations by reward
        ordered_generations = sorted(zip(texts, rewards), key=lambda x: x[1], reverse=True)
        
        # Add the best response to the new_responses list
        new_responses.append(ordered_generations[0][0])

    # Create the new dataset from the prompts and the new responses
    new_dataset = pd.DataFrame({'prompt': prompts, 'completion': new_responses})
    new_dataset.to_parquet(args.output_file + ".parquet", compression="gzip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nicholasKluge/seed-prompts")
    parser.add_argument("--split", default="english")
    parser.add_argument("--generation_model", default="nicholasKluge/Aira-2-124M")
    parser.add_argument("--reward_model", default="nicholasKluge/RewardModel")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--device_name", default="cuda")
    parser.add_argument("--output_file", default="rejection-fine-tuning-dataset")
    args = parser.parse_args()

    main(args)

# How to use:
# python create-rejection-fine-tuning-dataset.py --dataset_name "nicholasKluge/seed-prompts" --split "english" --generation_model "nicholasKluge/Aira-2-124M" --reward_model "nicholasKluge/RewardModel" --top_k 50 --max_length 500 --top_p 0.6 --temperature 0.6 --num_return_sequences 10 --device_name "cuda" --output_file "rejection-fine-tuning-dataset"

