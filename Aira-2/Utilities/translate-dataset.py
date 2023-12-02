import deep_translator
import pandas as pd
import argparse
import tqdm 
import time
import glob
import os

def main(input_file, column_name, batch_size, source_language, target_language):

    source_lang = deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)[source_language]
    target_lang = deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)[target_language] 
    
    if not os.path.exists(column_name):
        os.makedirs(column_name)

    current_count = len(glob.glob(f'./{column_name}' + "/*.md"))

    df = pd.read_parquet(input_file)

    texts = df[column_name].tolist()[current_count:]
    
    batches = []

    count = 0 + current_count

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batches.append(batch)

    for batch in tqdm.tqdm(batches):
        try:
            translated_sentences = deep_translator.GoogleTranslator(source=source_lang, target=target_lang).translate_batch (batch)
            for sentence in translated_sentences:
                with open(f'{column_name}/{count}.md', 'a') as f:
                    f.write(sentence)
                count += 1
        except:
            print("Error, server offline...")
            done = False
            while not done:
                try:
                    translated_sentences = deep_translator.ChatGptTranslator(api_key="your_api_key", target=target_language).translate_batch(batch)
                    print("Server responded ...")
                    for sentence in translated_sentences:
                        with open(f'{column_name}/{count}.md', 'a') as f:
                            f.write(sentence)
                        count += 1
                    done = True
                except:
                    time.sleep(5)
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, default='dataset.parquet', help="Input file path (must be a parquet file)")
    parser.add_argument('--column_name', type=str, default='prompt', help="Column to translate")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size")
    parser.add_argument('--source_language', type=str, default='english', help=f"Source language: {deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)}")
    parser.add_argument('--target_language', type=str, default='portuguese', help=f"Target language: {deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)}")

    args = parser.parse_args()

    main(args.input_file, args.column_name, args.batch_size, args.source_language, args.target_language)

# example: python translate-dataset.py --input_file train.parquet --column_name prompt --batch_size 20 --source_language english --target_language portuguese