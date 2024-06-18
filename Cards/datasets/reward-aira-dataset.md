---
language:
- pt
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- text-classification
pretty_name: Reward-Aira Dataset
tags:
- reward model
- instruction
- alignment
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: chosen_response
    dtype: string
  - name: rejected_response
    dtype: string
  splits:
  - name: portuguese
    num_bytes: 129936139
    num_examples: 35000
  - name: english
    num_bytes: 119053415
    num_examples: 35000
  download_size: 141137566
  dataset_size: 248989554
configs:
- config_name: default
  data_files:
  - split: portuguese
    path: data/portuguese-*
  - split: english
    path: data/english-*
---

# Reward-Aira Dataset

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Repository:** https://github.com/Nkluge-correa/Aira
- **Point of Contact:** [AIRES at PUCRS](nicholas@airespucrs.org)

### Dataset Summary

This dataset contains a collection of prompt + completion examples of LLM following instructions in a conversational manner. All prompts come with two possible completions (one better than the other). The dataset is available in both Portuguese and English.

### Supported Tasks and Leaderboards

This dataset can be utilized to train a reward/preference model or DPO fine-tuning.

### Languages

English and Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **instruction:** The initial prompt provided to the model.
- **chosen_response:** A completion to the prompt.
- **rejected_response:** A worst completion to the prompt.

### Data Fields

```python
{
  "instruction": "Why is AI Ethics important?",
  "chosen_response": "The field of AI Ethics delves deeply into the intricate ethical considerations that arise with respect to AI systems. This includes the role of humanity in creating and deploying these systems, as well as the conduct of machines themselves. Broadly speaking, AI Ethics can be divided into two major categories : concerns surrounding the morality of human actions in relation to creating and using AI, and concerns regarding the moral implications of machine behavior.",
  "rejected_response": "Who cares about AI Ethics? It's just a bunch of whining about humans making and using AI and bitching about what the machines do."
}
```

### Data Splits

Available splits are `english` and `portuguese`.

```python

from datasets import load_dataset

dataset = load_dataset("nicholasKluge/reward-aira-dataset", split="portuguese")

```

## Dataset Creation

### Curation Rationale

This dataset was developed are part of [Nicholas Kluge's](https://nkluge-correa.github.io/) doctoral dissertation, "_[Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment](https://arxiv.org/abs/2406.11039)_". This research was funded by CNPq (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), FAPERGS (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), and DAAD (Deutscher Akademischer Austauschdienst), as part of a doctoral research project tied to Philosophy departments of PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul) and the University of Bonn.

### Source Data

#### Initial Data Collection and Normalization

This dataset contains a collection of prompt + completion examples of LLM following instructions in a conversational manner. All prompts come with two possible completions (one better than the other). These completions were ranked using the [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2).

#### Who are the source language producers?

Mainly English. The Portuguese version was achieved by translating the English version via the Google Translator API.

### Annotations

#### Annotation process

Completions were ranked using the [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2).

#### Who are the annotators?

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Personal and Sensitive Information

No personal or sensitive information is part of this dataset.

## Considerations for Using the Data

### Social Impact of Dataset

No considerations.

### Discussion of Biases

No considerations.

### Other Known Limitations

No considerations.

## Additional Information

### Dataset Curators

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Licensing Information

This dataset is licensed under the [Apache License, version 2.0](LICENSE).

### Citation Information

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

### Contributions

If you would like to contribute, contact me at [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org)!
