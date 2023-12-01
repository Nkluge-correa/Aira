import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, help="Input file path (file should be in parquet format and have 'prompt' and 'completion' columns)")
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # fit the vectorizer on the prompt column
    prompt_tfidf_vectorizer = TfidfVectorizer()
    prompt_tfidf_vectorizer.fit(df['prompt'])

    # save the vectorizer
    joblib.dump(prompt_tfidf_vectorizer, args.output + 'prompt_vectorizer.pkl')

    # get the tfidf_matrix
    prompt_tfidf_matrix = prompt_tfidf_vectorizer.transform(df['prompt'])

    # save the tfidf_matrix
    joblib.dump(prompt_tfidf_matrix, args.output + 'prompt_tfidf_matrix.pkl')

    # fit the vectorizer on the completion column
    completion_tfidf_vectorizer = TfidfVectorizer()
    completion_tfidf_vectorizer.fit(df['completion'])

    # save the vectorizer
    joblib.dump(completion_tfidf_vectorizer, args.output + 'completion_vectorizer.pkl')

    # get the tfidf_matrix
    completion_tfidf_matrix = completion_tfidf_vectorizer.transform(df['completion'])

    # save the tfidf_matrix
    joblib.dump(completion_tfidf_matrix, args.output + 'completion_tfidf_matrix.pkl')

    print("Done!")

if __name__ == '__main__':
    main()

# example usage: python create-tfidf-matrix.py --input aira_instruct_english.parquet --output ./