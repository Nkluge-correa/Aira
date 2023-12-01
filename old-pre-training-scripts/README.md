# Legacy model pre-training

Here, you'll find the source code for the training of different types of language models with the `transformers`, `datasets`, and `torch` libraries. These are the first implementations we used to train such models from scratch (i.e., no fine-tuning). Results can be found on the [Hugging Face hub](https://huggingface.co/AiresPucrs/bert-base-bookcorpus).

## Usage 🕹️

- Train any of the pre defined models by running `python pre-training.py` on the `--spec-file` of your choise. You can also customize the `spec-files` settings in the YAML to suit your needs.

```bash
python pre-training.py --spec-file ./gpt2.yaml
```
