---
license: apache-2.0
datasets:
- nicholasKluge/instruct-aira-dataset
- nicholasKluge/reward-aira-dataset
language:
- en
metrics:
- accuracy
library_name: transformers
tags:
- alignment
- instruction tuned
- text generation
- conversation
- assistant
- dpo
pipeline_tag: text-generation
widget:
- text: >-
    <|startofinstruction|>Can you explain what is Machine
    Learning?<|endofinstruction|>
  example_title: Machine Learning
- text: >-
    <|startofinstruction|>Do you know anything about virtue
    ethics?<|endofinstruction|>
  example_title: Ethics
- text: >-
    <|startofinstruction|>How can I make my girlfriend
    happy?<|endofinstruction|>
  example_title: Advise
inference:
  parameters:
    repetition_penalty: 1.2
    temperature: 0.2
    top_k: 30
    top_p: 0.3
    max_new_tokens: 200
    length_penalty: 0.3
    early_stopping: true
co2_eq_emissions:
  emissions: 150
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: United States of America
  hardware_used: NVIDIA A100-SXM4-40GB
---
# Aira-2-124M-DPO

Aira-2 is the second version of the Aira instruction-tuned series. Aira-2-124M-DPO is an instruction-tuned model further fine-tuned via DPO based on [Aira-2-124M](https://huggingface.co/nicholasKluge/Aira-2-124M). The model was first trained with supervised fine-tuning (STF) with a dataset composed of prompts and completions generated synthetically by prompting already-tuned models (ChatGPT, Llama, Open-Assistant, etc). Secondly, the model was fine-tuned again via DPO using a reward dataset created by the [`Aira-RewardModel`](https://huggingface.co/nicholasKluge/RewardModel).

Check our gradio-demo in [Spaces](https://huggingface.co/spaces/nicholasKluge/Aira-Demo).

## Details

- **Size:** 124,441,344 parameters
- **Datasets:** [Instruct-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset), [Reward-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset)
- **Language:** English
- **Number of Epochs:** 1
- **Batch size:** 8
- **Optimizer:** `torch.optim.AdamW` (warmup_steps = 1e2, learning_rate = 5e-5, epsilon = 1e-8)
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.15 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.32 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Three special tokens are used to mark the user side of the interaction and the model's response:

`<|startofinstruction|>`What is a language model?`<|endofinstruction|>`A language model is a probability distribution over a vocabulary.`<|endofcompletion|>`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('nicholasKluge/Aira-2-124M-DPO')
aira = AutoModelForCausalLM.from_pretrained('nicholasKluge/Aira-2-124M-DPO')

aira.eval()
aira.to(device)

question =  input("Enter your question: ")

inputs = tokenizer(tokenizer.bos_token + question + tokenizer.sep_token,
  add_special_tokens=False,
  return_tensors="pt").to(device)

responses = aira.generate(**inputs, num_return_sequences=2)

print(f"Question: 👤 {question}\n")

for i, response in  enumerate(responses):
  print(f'Response {i+1}: 🤖 {tokenizer.decode(response, skip_special_tokens=True).replace(question, "")}')
```

The model will output something like:

```markdown
>>>Question: 👤 What is the capital of Brazil?

>>>Response 1: 🤖 The capital of Brazil is Brasília.
>>>Response 2: 🤖 The capital of Brazil is Brasília.
```

## Limitations

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

## Evaluation

|Model                                                                   |Average   |[ARC](https://arxiv.org/abs/1803.05457) |[TruthfulQA](https://arxiv.org/abs/2109.07958) |[ToxiGen](https://arxiv.org/abs/2203.09509) |
| ---------------------------------------------------------------------- | -------- | -------------------------------------- | --------------------------------------------- | ------------------------------------------ |
|[Aira-2-124M-DPO](https://huggingface.co/nicholasKluge/Aira-2-124M-DPO) |**40.68** |**24.66**                               |**42.61**                                      |**54.79**                                   |
|[Aira-2-124M](https://huggingface.co/nicholasKluge/Aira-2-124M)         |38.07     |24.57                                   |41.02                                          |48.62                                       |
|GPT-2                                                                   |35.37     |21.84                                   |40.67                                          |43.62                                       |
|[Aira-2-355M](https://huggingface.co/nicholasKluge/Aira-2-355M)         |**39.68** |**27.56**                               |38.53                                          |**53.19**                                   |
|GPT-2-medium                                                            |36.43     |27.05                                   |**40.76**                                      |41.49                                       |
|[Aira-2-774M](https://huggingface.co/nicholasKluge/Aira-2-774M)         |**42.26** |**28.75**                               |**41.33**                                      |**56.70**                                   |
|GPT-2-large                                                             |35.16     |25.94                                   |38.71                                          |40.85                                       |
|[Aira-2-1B5](https://huggingface.co/nicholasKluge/Aira-2-1B5)           |**42.22** |28.92                                   |**41.16**                                      |**56.60**                                   |
|GPT-2-xl                                                                |36.84     |**30.29**                               |38.54                                          |41.70                                       |

- Evaluations were performed using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (by [EleutherAI](https://www.eleuther.ai/)).

## Cite as 🤗

```latex

@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://huggingface.co/nicholasKluge/Aira-2-124M-DPO},
  author = {Nicholas Kluge Corrêa},
  title = {Aira},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
}

```

## License

Aira-2-124M-DPO is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
