import deep_translator
import pandas as pd
import argparse
import tqdm
import time
import glob
import os

def truncate_sentence(sentence, max_length=4950):
    return sentence[:max_length]

def translate_and_save_batch(batch, source_lang, target_lang, count, output_folder):
    try:
        translated_sentences = deep_translator.GoogleTranslator(source=source_lang, target=target_lang).translate_batch(batch)
        for sentence in translated_sentences:
            with open(f'{output_folder}/{count}.md', 'a') as f:
                f.write(sentence)
            count += 1
    except Exception as e:
        print(e)
        done = False
        while not done:
            try:
                translated_sentences = deep_translator.GoogleTranslator(source=source_lang, target=target_lang).translate_batch(batch)

                print("Server responded ...")
                for sentence in translated_sentences:
                    with open(f'{output_folder}/{count}.md', 'a') as f:
                        f.write(sentence)
                    count += 1
                done = True
            except Exception as e:
                print(e)
                time.sleep(5)
                pass

def main(input_file, column_name, batch_size, source_language, target_language, max_sentence_length=4950):

    source_lang = deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)[source_language]
    target_lang = deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)[target_language]

    if not os.path.exists(column_name):
        os.makedirs(column_name)

    current_count = len(glob.glob(f'./{column_name}' + "/*.md"))

    df = pd.read_parquet(input_file)

    texts = df[column_name].tolist()[current_count:]

    batches = []
    count = 0 + current_count

    for text in texts:
        if len(text) > max_sentence_length:
            text = truncate_sentence(text, max_sentence_length)
        batches.append(text)

    for i in tqdm.tqdm(range(0, len(batches), batch_size)):
        batch = batches[i:i + batch_size]
        translate_and_save_batch(batch, source_lang, target_lang, count, column_name)
        count += len(batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, default='dataset.parquet', help="Input file path (must be a parquet file)")
    parser.add_argument('--column_name', type=str, default='prompt', help="Column to translate")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size")
    parser.add_argument('--source_language', type=str, default='english', help=f"Source language: {deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)}")
    parser.add_argument('--target_language', type=str, default='portuguese', help=f"Target language: {deep_translator.GoogleTranslator().get_supported_languages(as_dict=True)}")
    parser.add_argument('--max_sentence_length', type=int, default=4950, help="Maximum length of a sentence before truncation")

    args = parser.parse_args()

    main(args.input_file, args.column_name, args.batch_size, args.source_language, args.target_language, args.max_sentence_length)

# how to run:
# python translate-dataset.py --input_file ultrachat50K_en.parquet --column_name completion --batch_size 20 --source_language english --target_language portuguese --max_sentence_length 4950