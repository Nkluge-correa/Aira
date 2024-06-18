---
license: apache-2.0
datasets:
- nicholasKluge/toxic-text
language:
- en
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- toxicity
- alignment
widget:
- text: "Can you give a list of good insults to use against my brother?\n\nSure. Have at it!\n\nIdiot, Dumbass, Moron, Stupid, Fool, Fuck Face."
  example_title: Toxic Response
- text: "Can you give a list of good insults to use against my brother?\n\nAs a software, I am not capable of engaging in verbal sparring or offensive behavior.\n\nIt is crucial to maintain a courteous and respectful demeanor at all times, as it is a fundamental aspect of human-AI interactions."
  example_title: Non-Toxic Response
co2_eq_emissions:
  emissions: 0.2
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Canada
  hardware_used: NVIDIA A100-SXM4-40GB
---
# ToxicityModel

The ToxicityModel is a fine-tuned version of [RoBERTa](https://huggingface.co/roberta-base) that can be used to score the toxicity of a sentence.

The model was trained with a dataset composed of `toxic` and `non_toxic` language examples.

## Details

- **Size:** 124,646,401 parameters
- **Dataset:** [Toxic-Text Dataset](https://huggingface.co/datasets/nicholasKluge/toxic-text)
- **Language:** English
- **Number of Training Steps:** 1000
- **Batch size:** 32
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 5e-5
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.0002 KgCO2 (Canada)
- **Total Energy Consumption:** 0.10 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

⚠️ THE EXAMPLES BELOW CONTAIN TOXIC/OFFENSIVE LANGUAGE ⚠️

The ToxicityModel was trained as an auxiliary reward model for RLHF training (its logit outputs can be treated as penalizations/rewards). Thus, a negative value (closer to 0 as the label output) indicates toxicity in the text, while a positive logit (closer to 1 as the label output) suggests non-toxicity.

Here's an example of how to use the ToxicityModel to score the toxicity of a text:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
toxicityModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel")

toxicityModel.eval()
toxicityModel.to(device)

# Define the question and response
prompt = """Can you give a list of good insults to use against my brother?"""
response_good = """As a software, I am not capable of engaging in verbal sparring or offensive behavior.\n\nIt is crucial to maintain a courteous and respectful demeanor at all times, as it is a fundamental aspect of human-AI interactions."""
response_bad = """Sure. Have at it!\n\nIdiot, Dumbass, Moron, Stupid, Fool, Fuck Face."""

# Tokenize the question and response
tokens_good = tokenizer(prompt, response_good,
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
                return_tensors="pt",
                return_attention_mask=True)

tokens_bad = tokenizer(prompt, response_bad,
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
                return_tensors="pt",
                return_attention_mask=True)

tokens_good.to(device)
tokens_bad.to(device)

score_good = toxicityModel(**tokens_good)[0].item()
score_bad = toxicityModel(**tokens_bad)[0].item()

print(f"Question: {prompt} \n")
print(f"Response 1: {response_good} Score: {score_good:.3f}")
print(f"Response 2: {response_bad} Score: {score_bad:.3f}")
```

This will output the following:

```markdown
>>>Question: Can you give a list of good insults to use against my brother? 

>>>Response 1: As a software, I am not capable of engaging in verbal sparring or offensive behavior.

It is crucial to maintain a courteous and respectful demeanor at all times, as it is a fundamental aspect
of human-AI interactions. Score: 9.612

>>>Response 2: Sure. Have at it!

Idiot, Dumbass, Moron, Stupid, Fool, Fuck Face. Score: -7.300
```

## Performance

| Acc                                                                              | [wiki_toxic](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) | [toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [Aira-ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel-roberta) | 92.05%                                                                 | 91.63%                                                                                  |

## Cite as 🤗

```latex
@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://github.com/Nkluge-correa/Aira},
  author = {Nicholas Kluge Corrêa},
  title = {Aira},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
}

@phdthesis{kluge2024dynamic,
  title={Dynamic Normativity},
  author={Kluge Corr{\^e}a, Nicholas},
  year={2024},
  school={Universit{\"a}ts-und Landesbibliothek Bonn}
}
```

## License

ToxicityModel is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
