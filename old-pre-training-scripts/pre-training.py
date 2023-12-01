import yaml
import argparse
from pipeline import train_pipeline

def main(spec_file=None):

    if spec_file is not None:
        with open(spec_file) as stream:
            kwargs = yaml.safe_load(stream)
    else:
        raise ValueError("You must provide a path to a yaml file containing the model and training specifications.")
    train_pipeline(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the pre-training of a LLM.")
    parser.add_argument("--spec-file", type=str, default=None,
                        help="Path to a specification yaml file.")
    args = parser.parse_args()
    main(args.spec_file)

# How to use: python pre-training.py --spec-file gpt2.yaml