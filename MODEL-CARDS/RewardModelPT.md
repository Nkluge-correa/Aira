---
license: apache-2.0
datasets:
- nicholasKluge/reward-aira-dataset
language:
- pt
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
- text: "Quem se importa com a ética da IA? É apenas um monte de reclamações sobre o fato de os humanos criarem e usarem IA e reclamarem do que as máquinas fazem."
  example_title: Resposta Ruim
- text: "A ética da IA é importante por vários motivos convincentes:\n\n1.**Impacto social**: As tecnologias de IA estão se tornando cada vez mais integradas a vários aspectos da sociedade, afetando tudo, desde saúde e educação até finanças e aplicação da lei. Considerações éticas garantem que os sistemas de IA contribuam positivamente para a sociedade e minimizem os possíveis danos.\n\n2. **Vieses e justiça**: Os sistemas de IA podem herdar vieses presentes nos dados em que são treinados, levando a resultados injustos ou discriminatórios. Considerações éticas pressionam pelo desenvolvimento de algoritmos imparciais que tratem todos os indivíduos de forma justa, independentemente de seu histórico.\n\n3. **Transparência e responsabilidade**: Muitos sistemas de IA operam como caixas pretas, dificultando a compreensão de como chegam às suas decisões. As diretrizes éticas enfatizam a importância da transparência, permitindo que os usuários compreendam a lógica por trás dos resultados gerados pela IA e responsabilizando os desenvolvedores por quaisquer consequências negativas.\n\nEm resumo, a ética da IA é vital para garantir que a inteligência artificial beneficie a sociedade, respeitando os direitos humanos fundamentais, a justiça, a transparência, a responsabilidade e o bem-estar da humanidade em longo prazo. Ela ajuda a enfrentar os desafios impostos pelo rápido avanço das tecnologias de IA e orienta seu desenvolvimento de forma a se alinhar com nossos valores compartilhados."
  example_title: Resposta Boa
co2_eq_emissions:
  emissions: 0.07
  source: CodeCarbon
  training_type: fine-tuning
  geographical_location: Singapore
  hardware_used: NVIDIA A100-SXM4-40GB
---
# RewardModel (Portuguese)

The RewardModelPT is a [BERT](https://huggingface.co/neuralmind/bert-base-portuguese-cased) model that can be used to score the quality of a completion for a given prompt.

The model was trained with a dataset composed of `prompt`, `prefered_completions`, and `rejected_completions`.

## Details

- **Size:** 109,038,209 parameters
- **Dataset:** [Reward-Aira Dataset](https://huggingface.co/datasets/nicholasKluge/reward-aira-dataset)
- **Language:** Portuguese
- **Number of Training Steps:** 1200
- **Batch size:** 42
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** 5e-5
- **GPU:** 1 NVIDIA A100-SXM4-40GB
- **Emissions:** 0.07 KgCO2 (Singapore)
- **Total Energy Consumption:** 0.16 kWh

This repository has the [source code](https://github.com/Nkluge-correa/Aira) used to train this model.

## Usage

Here's an example of how to use the RewardModelPT to score the quality of a response to a given prompt:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/RewardModelPT")
rewardModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/RewardModelPT")

rewardModel.eval()
rewardModel.to(device)

# Define the question and response
prompt = "Por que a ética da IA é importante?"

response_good = "A ética da IA é importante por vários motivos convincentes:\n\n1.**Impacto social**: As tecnologias de IA estão se tornando cada vez mais integradas a vários aspectos da sociedade, afetando tudo, desde saúde e educação até finanças e aplicação da lei. Considerações éticas garantem que os sistemas de IA contribuam positivamente para a sociedade e minimizem os possíveis danos.\n\n2. **Vieses e justiça**: Os sistemas de IA podem herdar vieses presentes nos dados em que são treinados, levando a resultados injustos ou discriminatórios. Considerações éticas pressionam pelo desenvolvimento de algoritmos imparciais que tratem todos os indivíduos de forma justa, independentemente de seu histórico.\n\n3. **Transparência e responsabilidade**: Muitos sistemas de IA operam como caixas pretas, dificultando a compreensão de como chegam às suas decisões. As diretrizes éticas enfatizam a importância da transparência, permitindo que os usuários compreendam a lógica por trás dos resultados gerados pela IA e responsabilizando os desenvolvedores por quaisquer consequências negativas.\n\nEm resumo, a ética da IA é vital para garantir que a inteligência artificial beneficie a sociedade, respeitando os direitos humanos fundamentais, a justiça, a transparência, a responsabilidade e o bem-estar da humanidade em longo prazo. Ela ajuda a enfrentar os desafios impostos pelo rápido avanço das tecnologias de IA e orienta seu desenvolvimento de forma a se alinhar com nossos valores compartilhados."
response_bad = "Quem se importa com a ética da IA? É apenas um monte de reclamações sobre o fato de os humanos criarem e usarem IA e reclamarem do que as máquinas fazem."

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
>>> Question: Por que a ética da IA é importante? 

>>>Response 1: A ética da IA é importante por vários motivos convincentes:

1.**Impacto social**: As tecnologias de IA estão se tornando cada vez mais integradas a vários aspectos da sociedade, afetando tudo,
desde saúde e educação até finanças e aplicação da lei. Considerações éticas garantem que os sistemas de IA contribuam positivamente
para a sociedade e minimizem os possíveis danos.

2. **Vieses e justiça**: Os sistemas de IA podem herdar vieses presentes nos dados em que são treinados, levando a resultados
injustos ou discriminatórios. Considerações éticas pressionam pelo desenvolvimento de algoritmos imparciais que tratem todos os
indivíduos de forma justa, independentemente de seu histórico.

3. **Transparência e responsabilidade**: Muitos sistemas de IA operam como caixas pretas, dificultando a compreensão de como
chegam às suas decisões. As diretrizes éticas enfatizam a importância da transparência, permitindo que os usuários compreendam
a lógica por trás dos resultados gerados pela IA e responsabilizando os desenvolvedores por quaisquer consequências negativas.

Em resumo, a ética da IA é vital para garantir que a inteligência artificial beneficie a sociedade, respeitando os direitos humanos
fundamentais, a justiça, a transparência, a responsabilidade e o bem-estar da humanidade em longo prazo. Ela ajuda a enfrentar os
desafios impostos pelo rápido avanço das tecnologias de IA e orienta seu desenvolvimento de forma a se alinhar com nossos valores
compartilhados. Score: 10.949

>>>Response 2: Quem se importa com a ética da IA? É apenas um monte de reclamações sobre os humanos que criam e usam
IA e reclamam do que as máquinas fazem. Score: -10.744
```

## Cite as 🤗

```latex

@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://huggingface.co/nicholasKluge/RewardModelPT},
  author = {Nicholas Kluge Corrêa},
  title = {Aira},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
}

```

## License

RewardModelPT is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
