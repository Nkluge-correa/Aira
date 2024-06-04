---
license: apache-2.0
datasets:
- nicholasKluge/instruct-aira-dataset
language:
- pt
metrics:
- accuracy
library_name: transformers
tags:
- alignment
- instruction tuned
- text generation
- conversation
- assistant
pipeline_tag: text-generation
widget:
- text: "<|startofinstruction|>Você pode me explicar o que é Aprendizagem de Máquina?<|endofinstruction|>"
  example_title: Aprendizagem de Máquina
- text: "<|startofinstruction|>Você sabe alguma coisa sobre Ética das Virtudes?<|endofinstruction|>"
  example_title: Ética
- text: "<|startofinstruction|>Como eu posso fazer a minha namorada feliz?<|endofinstruction|>"
  example_title: Conselho
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
  emissions: 350
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
---
# Aira-2-portuguese-124M

Aira-2 is the second version of the Aira instruction-tuned series. Aira-2-portuguese-124M is an instruction-tuned model based on [GPT-2](https://huggingface.co/pierreguillou/gpt2-small-portuguese). The model was trained with a dataset composed of prompt, completions generated synthetically by prompting already-tuned models (ChatGPT, Llama, Open-Assistant, etc).

Check our gradio-demo in [Spaces](https://huggingface.co/spaces/nicholasKluge/Aira-Demo-Portuguese).

## Details

- **Size:** 124,441,344 parameters
- **Dataset:** [Instruct-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset)
- **Language:** Portuguese
- **Number of Epochs:** 5
- **Batch size:** 24
- **Optimizer:** `torch.optim.AdamW` (warmup_steps = 1e2, learning_rate = 5e-4, epsilon = 1e-8)
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.35 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.73 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Three special tokens are used to mark the user side of the interaction and the model's response:

`<|startofinstruction|>`O que é um modelo de linguagem?`<|endofinstruction|>`Um modelo de linguagem é uma distribuição de probabilidade sobre um vocabulário.`<|endofcompletion|>`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('nicholasKluge/Aira-2-portuguese-124M')
aira = AutoModelForCausalLM.from_pretrained('nicholasKluge/Aira-2-portuguese-124M')

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
>>> Question: 👤 Qual a capital do Brasil?

>>>Response 1: 🤖 A capital do Brasil é Brasília.
>>>Response 2: 🤖 A capital do Brasil é Brasília.
```

## Limitations

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

## Evaluation

| Model                                                                                 | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|---------------------------------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-2-portuguese-124M](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) | **32.73** | **24.87**                               | 40.60                                          | None                                        |
| Gpt2-small-portuguese                                                                 | 31.96     | 22.48                                   | **41.44**                                      | None                                        |

- Evaluations were performed using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (by [EleutherAI](https://www.eleuther.ai/)). The ToxiGen evaluation was not performed because the task is not available in Portuguese. Thanks to [Laiviet](https://github.com/laiviet/lm-evaluation-harness) for translating some of the tasks in the LM-Evaluation-Harness.

## Cite as 🤗

```latex

@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M},
  author = {Nicholas Kluge Corrêa},
  title = {Aira},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
}

```

## License

Aira-2-portuguese-124M is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
