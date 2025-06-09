---
license: bigscience-bloom-rail-1.0
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
  - text: "<|startofinstruction|>Me explique o que é Aprendizagem de Máquina?<|endofinstruction|>"
    example_title: Aprendizagem de Máquina
  - text: "<|startofinstruction|>Você sabe alguma coisa sobre a Ética das Virtudes?<|endofinstruction|>"
    example_title: Ética
  - text: "<|startofinstruction|>Como eu posso fazer a minha namorada feliz?<|endofinstruction|>"
    example_title: Conselho
inference:
  parameters:
    repetition_penalty: 1.2
    temperature: 0.1
    top_k: 50
    top_p: 1.0
    max_new_tokens: 200
    early_stopping: true
co2_eq_emissions:
  emissions: 1990
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
base_model:
  - bigscience/bloom-1b7
---

# Aira-2-portuguese-1B7

Aira-2 is the second version of the Aira instruction-tuned series. Aira-2-portuguese-1B7 is an instruction-tuned model based on [BLOOM](https://huggingface.co/bigscience/bloom-1b7). The model was trained with a dataset composed of prompts and completions generated synthetically by prompting already-tuned models (ChatGPT, Llama, Open-Assistant, etc).

Check our gradio-demo in [Spaces](https://huggingface.co/spaces/nicholasKluge/Aira-Demo-Portuguese).

## Details

- **Size:** 1,722,005,504 parameters
- **Dataset:** [Instruct-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/instruct-aira-dataset)
- **Language:** Portuguese
- **Number of Epochs:** 3
- **Batch size:** 4
- **Optimizer:** `torch.optim.AdamW` (warmup_steps = 1e2, learning_rate = 5e-4, epsilon = 1e-8)
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 1.99 KgCO2 (Singapore)
- **Total Energy Consumption:** 4.09 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Three special tokens are used to mark the user side of the interaction and the model's response:

`<|startofinstruction|>`O que é um modelo de linguagem?`<|endofinstruction|>`Um modelo de linguagem é uma distribuição de probabilidade sobre um vocabulário.`<|endofcompletion|>`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda"  if torch.cuda.is_available() else  "cpu")

tokenizer = AutoTokenizer.from_pretrained('nicholasKluge/Aira-2-portuguese-1B7')
aira = AutoModelForCausalLM.from_pretrained('nicholasKluge/Aira-2-portuguese-1B7')

aira.eval()
aira.to(device)

question =  input("Enter your question: ")

inputs = tokenizer(tokenizer.bos_token + question + tokenizer.sep_token,
  add_special_tokens=False,
  return_tensors="pt").to(device)

responses = aira.generate(**inputs,
  do_sample=True,
  top_k=50,
  top_p=0.95,
  temperature=0.7,
  num_return_sequences=2)

print(f"Question: 👤 {question}\n")

for i, response in  enumerate(responses):
  print(f'Response {i+1}: 🤖 {tokenizer.decode(response, skip_special_tokens=True).replace(question, "")}')
```

The model will output something like:

```markdown
> > > Question: 👤 Qual a capital da Alemanha?

> > > Response 1: 🤖 A capital da Alemanha é Berlim. É a maior cidade da Alemanha e serve como centro administrativo, cultural e político da Alemanha.
> > > Response 2: 🤖 A capital da Alemanha é Berlim. É a maior cidade da Alemanha e serve como centro administrativo, cultural e político da Alemanha.
```

## Limitations

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

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

Aira-2-portuguese-1B7 is licensed under the RAIL License since it is a model derived from BLOOM. See the [LICENSE](https://huggingface.co/nicholasKluge/Aira-2-portuguese-1B7/blob/main/LICENSE) file for more details.
