import pandas as pd
import argparse
import openai
import tqdm 
import time
import glob
import os

openai.api_key="your_api_key"

def main(model, input_file, output_dir, max_tokens):
    """
    Create a dataset of completions for a given input file and model

    Args:
        model (str): OpenAI model to use
        input_file (str): Input file path
        output_dir (str): Output directory path

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_count = len(glob.glob(f'./{output_dir}' + "/*.txt"))

    df = pd.read_parquet(input_file)

    prompts = df.prompt.tolist()[current_count:]

    system_prompt = input("Enter system prompt: ")

    for i in tqdm.tqdm(range(len(prompts))):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompts[i]},
                ],
                max_tokens=max_tokens,
            )
            with open(f'{output_dir}/{i+current_count}.txt', 'a') as f:
                f.write(response['choices'][0]['message']['content'])
        except:
            print("Error, server offline...")
            done = False
            while not done:
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompts[i]},
                        ],
                        max_tokens=max_tokens,
                    )
                    print("Server responded ...")
                    with open(f'{output_dir}/{i+current_count}.txt', 'a') as f:
                        f.write(response['choices'][0]['message']['content'])
                    done = True
                except:
                    time.sleep(5)
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI model to use (gpt-4)')
    parser.add_argument('--input_file', type=str, default='dataset.parquet', help="Input file path (must be a parquet file with the column 'prompt')")
    parser.add_argument('--output_dir', type=str, default='completions', help='Output directory path (default: completions)')
    parser.add_argument('--max_tokens', type=int, default=325, help='Max tokens for completion')

    args = parser.parse_args()

    main(args.model, args.input_file, args.output_dir, args.max_tokens)

# example: python create-instruction-tuning-dataset.py --model gpt-4 --input_file dataset.parquet --output_dir completions --max_tokens 325