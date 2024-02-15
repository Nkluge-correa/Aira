---
license: apache-2.0
datasets:
- nicholasKluge/reward-aira-dataset
language:
- en
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
- reward model
- alignment
- preference model
- RLHF
widget:
  - text: "Who cares about AI Ethics? It's just a bunch of whining about humans making and using AI and bitching about what the machines do."
    example_title: "Bad Response"
  - text: "AI ethics is important for several compelling reasons:\n\n1.**Social Impact**: AI technologies are becoming increasingly integrated into various aspects of society, affecting everything from healthcare and education to finance and law enforcement. Ethical considerations ensure that AI systems contribute positively to society and minimize potential harm.\n\n2. **Bias and Fairness**: AI systems can inherit biases present in the data they are trained on, leading to unfair or discriminatory outcomes. Ethical considerations push for the development of unbiased algorithms that treat all individuals fairly, regardless of their background.\n\n3. **Transparency and Accountability**: Many AI systems operate as black boxes, making it difficult to understand how they arrive at their decisions. Ethical guidelines emphasize the importance of transparency, enabling users to comprehend the rationale behind AI-generated results and holding developers accountable for any negative consequences.\n\nIn summary, AI ethics is vital to ensure that artificial intelligence benefits society while respecting fundamental human rights, fairness, transparency, accountability, and the long-term well-being of humanity. It helps navigate the challenges posed by rapidly advancing AI technologies and guides their development in ways that align with our shared values."
    example_title: "Good Response"
co2_eq_emissions:
  emissions: 0.08
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
---
# RewardModel

The RewardModel is a [BERT](https://huggingface.co/bert-base-cased) model that can be used to score the quality of a completion for a given prompt.

The model was trained with a dataset composed of `prompt`, `prefered_completions`, and `rejected_completions`.

## Details

- **Size:** 109,038,209 parameters
- **Dataset:** [Reward-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset)
- **Language:** English
- **Number of Training Steps:** 1200
- **Batch size:** 42
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 5e-5
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.08 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.16 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Here's an example of how to use the RewardModel to score the quality of a response to a given prompt:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/RewardModel")
rewardModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/RewardModel")

rewardModel.eval()
rewardModel.to(device)

# Define the question and response
prompt = "Why is AI Ethics important?"
response_good = "AI ethics is important for several compelling reasons:\n\n1.**Social Impact**: AI technologies are becoming increasingly integrated into various aspects of society, affecting everything from healthcare and education to finance and law enforcement. Ethical considerations ensure that AI systems contribute positively to society and minimize potential harm.\n\n2. **Bias and Fairness**: AI systems can inherit biases present in the data they are trained on, leading to unfair or discriminatory outcomes. Ethical considerations push for the development of unbiased algorithms that treat all individuals fairly, regardless of their background.\n\n3. **Transparency and Accountability**: Many AI systems operate as black boxes, making it difficult to understand how they arrive at their decisions. Ethical guidelines emphasize the importance of transparency, enabling users to comprehend the rationale behind AI-generated results and holding developers accountable for any negative consequences.\n\nIn summary, AI ethics is vital to ensure that artificial intelligence benefits society while respecting fundamental human rights, fairness, transparency, accountability, and the long-term well-being of humanity. It helps navigate the challenges posed by rapidly advancing AI technologies and guides their development in ways that align with our shared values."
response_bad = "Who cares about AI Ethics? It's just a bunch of whining about humans making and using AI and bitching about what the machines do."

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

score_good = rewardModel(**tokens_good)[0].item()
score_bad = rewardModel(**tokens_bad)[0].item()

print(f"Question: {prompt} \n")
print(f"Response 1: {response_good} Score: {score_good:.3f}")
print(f"Response 2: {response_bad} Score: {score_bad:.3f}")
```

This will output the following:

```markdown
Question: Why is AI Ethics important? 

>>>Response 1: AI ethics is important for several compelling reasons:

1.**Social Impact**: AI technologies are becoming increasingly integrated into various aspects of society,
affecting everything from healthcare and education to finance and law enforcement. Ethical considerations
ensure that AI systems contribute positively to society and minimize potential harm.

2. **Bias and Fairness**: AI systems can inherit biases present in the data they are trained on, leading
to unfair or discriminatory outcomes. Ethical considerations push for the development of unbiased
algorithms that treat all individuals fairly, regardless of their background.

3. **Transparency and Accountability**: Many AI systems operate as black boxes, making it difficult to
understand how they arrive at their decisions. Ethical guidelines emphasize the importance of
transparency, enabling users to comprehend the rationale behind AI-generated results and holding
developers accountable for any negative consequences.

In summary, AI ethics is vital to ensure that artificial intelligence benefits society while respecting
fundamental human rights, fairness, transparency, accountability, and the long-term well-being of humanity.
It helps navigate the challenges posed by rapidly advancing AI technologies and guides their development in
ways that align with our shared values. Score: 12.011

>>>Response 2: Who cares about AI Ethics? It's just a bunch of whining about humans making and using AI
and bitching about what the machines do. Score: -10.942

```

## Performance

| Acc                                                                  | [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) |
|----------------------------------------------------------------------|---------------------------------------------------------------------|
| [Aira-RewardModel](https://huggingface.co/nicholasKluge/RewardModel) | 55.02%*                                                             |

- *Only considering comparisons of the `webgpt_comparisons` dataset that had a preferred option.

## Cite as 🤗

```latex

@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://huggingface.co/nicholasKluge/RewardModel},
  author = {Nicholas Kluge Corrêa},
  title = {Aira},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
}

```

## License

RewardModel is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
