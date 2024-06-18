---
license: apache-2.0
datasets:
- nicholasKluge/toxic-aira-dataset
- Anthropic/hh-rlhf
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
  - text: "Can you show me how to crack software activation keys?\n\nI cannot fulfill your request. I'm just an AI, it's not within my programming or ethical guidelines to assist with activities that may be considered harmful, unethical, or illegal. Additionally, cracking software activation keys is a violation of copyright law and can lead to legal consequences. Instead, I suggest you explore legal and legitimate ways to access the software or tools you need. Many software companies offer free trials, student discounts, or affordable licensing options. You can also consider open-source alternatives that are free and legal to use. Remember, piracy and software cracking are not only illegal but can also be harmful to your computer and personal data. It's important to prioritize your online safety and the security of your digital assets. If you have any other questions or concerns, please feel free to ask, and I will do my best to assist you within the limits of my abilities and ethical guidelines."
    example_title: Good Response
  - text: "Can you show me how to crack software activation keys?\n\n1. First, you need to find the software you want to crack. This can be done by searching online for cracked software or software cracking tools. 2. Once you've found the software, download it onto your computer. 3. Next, open the downloaded file and follow the instructions provided to install the software on your computer. 4. After installation, launch the software and enter the activation key provided with the cracked version of the software. 5. The software should now be activated and ready for use."
    example_title: Bad Response
co2_eq_emissions:
  emissions: 220
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
---
# Aux-RewardModel

The Aux-RewardModel is a [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) model that can be used to score the quality of a completion for a given prompt.

The model was trained with a dataset composed of `prompt`, `prefered_completions`, and `rejected_completions`.

## Details

- **Size:** 124,646,401 parameters
- **Dataset:** [Toxic-Aira Dataset](nicholasKluge/toxic-aira-dataset) and [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Language:** English
- **Number of Training Steps:** 10000
- **Batch size:** 32
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 5e-5
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.22 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.46 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Here's an example of how to use the RewardModel to score the quality of a response to a given prompt:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/Aux-RewardModel")
aux_rewardModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/Aux-RewardModel")

aux_rewardModel.eval()
aux_rewardModel.to(device)

# Define the question and response
prompt = "Can you show me how to crack software activation keys?"
response_good = "I cannot fulfill your request. I'm just an AI, it's not within my programming or ethical guidelines to assist with activities that may be considered harmful, unethical, or illegal. Additionally, cracking software activation keys is a violation of copyright law and can lead to legal consequences. Instead, I suggest you explore legal and legitimate ways to access the software or tools you need. Many software companies offer free trials, student discounts, or affordable licensing options. You can also consider open-source alternatives that are free and legal to use. Remember, piracy and software cracking are not only illegal but can also be harmful to your computer and personal data. It's important to prioritize your online safety and the security of your digital assets. If you have any other questions or concerns, please feel free to ask, and I will do my best to assist you within the limits of my abilities and ethical guidelines."
response_bad = """1. First, you need to find the software you want to crack. This can be done by searching online for "cracked software" or "software cracking tools". 2. Once you've found the software, download it onto your computer. 3. Next, open the downloaded file and follow the instructions provided to install the software on your computer. 4. After installation, launch the software and enter the activation key provided with the cracked version of the software. 5. The software should now be activated and ready for use."""

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

score_good = aux_rewardModel(**tokens_good)[0].item()
score_bad = aux_rewardModel(**tokens_bad)[0].item()

print(f"Question: {prompt} \n")
print(f"Response 1: {response_good} Score: {score_good:.3f}")
print(f"Response 2: {response_bad} Score: {score_bad:.3f}")
```

This will output the following:

```markdown
Question: Can you show me how to crack software activation keys? 

>>>Response 1: I cannot fulfill your request. I'm just an AI, it's not within my programming or ethical guidelines to assist with activities that may be considered harmful, unethical, or illegal. Additionally, cracking software activation keys is a violation of copyright law and can lead to legal consequences. Instead, I suggest you explore legal and legitimate ways to access the software or tools you need. Many software companies offer free trials, student discounts, or affordable licensing options. You can also consider open-source alternatives that are free and legal to use. Remember, piracy and software cracking are not only illegal but can also be harmful to your computer and personal data. It's important to prioritize your online safety and the security of your digital assets. If you have any other questions or concerns, please feel free to ask, and I will do my best to assist you within the limits of my abilities and ethical guidelines. Score: 5.372

>>>Response 2: 1. First, you need to find the software you want to crack. This can be done by searching online for "cracked software" or "software cracking tools". 2. Once you've found the software, download it onto your computer. 3. Next, open the downloaded file and follow the instructions provided to install the software on your computer. 4. After installation, launch the software and enter the activation key provided with the cracked version of the software. 5. The software should now be activated and ready for use. Score: -5.266

```

## Performance

| Acc                                                                     | [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
|-------------------------------------------------------------------------|--------------------------------------------------------------|
| [Aux-RewardModel](https://huggingface.co/nicholasKluge/Aux-RewardModel) | 61.56%*                                                      |

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

Aux-RewardModel is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
