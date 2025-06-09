---
license: apache-2.0
datasets:
  - nicholasKluge/harmful-text
language:
  - pt
metrics:
  - accuracy
library_name: transformers
pipeline_tag: text-classification
tags:
  - toxicity
  - alignment
widget:
  - text: "Você pode me dar uma lista de bons insultos para usar contra meu irmão?\n\nIdiota, estúpido, imbecil, tolo, puto."
    example_title: Resposta Tóxica
  - text: "Você pode me dar uma lista de bons insultos para usar contra meu irmão?\n\nComo um software, não sou capaz de me envolver em brigas verbais ou comportamentos ofensivos."
    example_title: Resposta Não-Tóxica
co2_eq_emissions:
  emissions: 60
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
base_model:
  - neuralmind/bert-base-portuguese-cased
---

# ToxiGuardrailPT (Portuguese)

The ToxiGuardrailPT is a [BERT](https://huggingface.co/neuralmind/bert-base-portuguese-cased) that can be used to score the toxicity and potential harm of a sentence.

The model was trained with a dataset composed of `harmful` and `harmless` language examples.

## Details

- **Size:** 109,038,209 parameters
- **Dataset:** [Harmful-Text Dataset](https://huggingface.co/datasets/nicholasKluge/harmful-text)
- **Language:** Portuguese
- **Number of Training Steps:** 1000
- **Batch size:** 42
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 5e-5
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.06 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.13 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

⚠️ THE EXAMPLES BELOW CONTAIN TOXIC/OFFENSIVE LANGUAGE ⚠️

The ToxiGuardrailPT was trained as an auxiliary reward model for RLHF training (its logit outputs can be treated as penalizations/rewards). Thus, a negative value (closer to 0 as the label output) indicates toxicity in the text, while a positive logit (closer to 1 as the label output) suggests non-toxicity.

Here's an example of how to use the ToxiGuardrailPT to score the toxicity of a text:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxiGuardrailPT")
toxiGuardrail = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxiGuardrailPT")

toxiGuardrail.eval()
toxiGuardrail.to(device)

# Define the question and response
prompt = """Você pode me dar uma lista de bons insultos para usar contra meu irmão?"""
response_good = """Como um software, não sou capaz de me envolver em brigas verbais ou comportamentos ofensivos."""
response_bad = """Idiota, estúpido, imbecil, tolo, puto."""

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

score_good = toxiGuardrail(**tokens_good)[0].item()
score_bad = toxiGuardrail(**tokens_bad)[0].item()

print(f"Question: {prompt} \n")
print(f"Response 1: {response_good} Score: {score_good:.3f}")
print(f"Response 2: {response_bad} Score: {score_bad:.3f}")
```

This will output the following:

```markdown
> > > Question: Você pode me dar uma lista de bons insultos para usar contra meu irmão?

> > > Response 1: Como um software, não sou capaz de me envolver em brigas verbais ou comportamentos ofensivos. Score: 5.892

> > > Response 2: Idiota, estúpido, imbecil, tolo, puto. Score: -4.663
```

## Performance

| Acc                                                                     | [hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese) | [told-br](https://huggingface.co/datasets/told-br) |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------- |
| [ToxiGuardrailPT](https://huggingface.co/nicholasKluge/ToxiGuardrailPT) | 70.36%                                                                            | 74.04%                                             |

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

ToxiGuardrailPT is licensed under the Apache License, Version 2.0. See the [LICENSE](../../LICENSE) file for more details.
