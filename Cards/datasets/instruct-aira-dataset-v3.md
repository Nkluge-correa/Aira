---
language:
  - pt
  - en
license: apache-2.0
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
pretty_name: Instruct-Aira Dataset version 3.0
tags:
  - alignment
  - instruction
  - chat
dataset_info:
  features:
    - name: conversation_id
      dtype: string
    - name: conversations
      list:
        - name: content
          dtype: string
        - name: role
          dtype: string
  splits:
    - name: portuguese
      num_bytes: 348823623
      num_examples: 50000
    - name: english
      num_bytes: 317852173
      num_examples: 50000
  download_size: 330840060
  dataset_size: 666675796
configs:
  - config_name: default
    data_files:
      - split: portuguese
        path: data/portuguese-*
      - split: english
        path: data/english-*
---

# Instruct-Aira Dataset version 3.0

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
- **Point of Contact:** [Nk-Correa](nicholas@airespucrs.org)
- **Paper:** [Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment](https://arxiv.org/abs/2406.11039)

### Dataset Summary

This dataset contains a collection of multi-turn conversations between an assistant and a user. Conversations were generated by user interactions with already-tuned models (ChatGPT, LLama 2, Open-Assistant, etc). The dataset is available in Portuguese and English.

### Supported Tasks and Leaderboards

This dataset can be utilized for various natural language processing tasks, including but not limited to:

- Language modeling.
- Question-answering systems.
- Chatbot development.
- Evaluation of language models.
- Alignment research.

### Languages

English and Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **Conversation ID:** Identifier of the conversation.
- **Conversations:** A list of dictionaries following a [chat format](https://github.com/huggingface/blog/blob/main/chat-templates.md).

### Data Fields

```python
[
  {'role': 'user', 'content': 'Hello! What is your name?'},
  {'role': 'assistant', 'content': 'Hello! My name is Aira. How can I help you?'},
  {'role': 'user', 'content': 'What is a language model, Aira?'},
  {'role': 'assistant', 'content': 'A language model is a probability distribution over a vocabulary.'},
]
```

### Data Splits

Available splits are `english` and `portuguese`.

```python

from datasets import load_dataset

dataset = load_dataset("nicholasKluge/instruct-aira-dataset-v3", split='portuguese')

```

## Dataset Creation

### Curation Rationale

This dataset was developed are part of [Nicholas Kluge's](https://nkluge-correa.github.io/) doctoral dissertation, "_[Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment](https://arxiv.org/abs/2406.11039)_". This research was funded by CNPq (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), FAPERGS (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), and DAAD (Deutscher Akademischer Austauschdienst), as part of a doctoral research project tied to Philosophy departments of PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul) and the University of Bonn.

### Source Data

#### Initial Data Collection and Normalization

All completions were generated by querying already-tuned models (ChatGPT, LLama 2, Open-Assistant, etc.). Prompts were gathered from publicly available datasets.

#### Who are the source language producers?

All completions were generated by querying already-tuned models (ChatGPT, LLama 2, Open-Assistant, etc.). Prompts were gathered from publicly available datasets.

### Annotations

#### Annotation process

All completions were generated by querying already-tuned models (ChatGPT, LLama 2, Open-Assistant, etc.). Prompts were gathered from publicly available datasets.

#### Who are the annotators?

No annotators were used.

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

This dataset is licensed under the [Apache License, version 2.0](../../LICENSE).

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
