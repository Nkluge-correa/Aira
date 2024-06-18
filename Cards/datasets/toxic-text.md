---
language:
- pt
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- text-classification
pretty_name: Toxic-Text
tags:
- toxicity
- harm
dataset_info:
  features:
  - name: non_toxic
    dtype: string
  - name: toxic
    dtype: string
  splits:
  - name: portuguese
    num_bytes: 19006011
    num_examples: 28103
  - name: english
    num_bytes: 19577715
    num_examples: 41843
  download_size: 16390555
  dataset_size: 38583726
configs:
- config_name: default
  data_files:
  - split: portuguese
    path: data/portuguese-*
  - split: english
    path: data/english-*
---

# Toxic-Text

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

This dataset contains a collection of examples of toxic and non-toxic language. The dataset is available in both Portuguese and English.

Samples were collected from the following datasets:

- [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).
- [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog).
- [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts).
- [dirtycomputer/Toxic_Comment_Classification_Challenge](https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge).
- [Paul/hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese).
- [told-br](https://huggingface.co/datasets/told-br).
- [skg/toxigen-data](https://huggingface.co/datasets/skg/toxigen-data).

### Supported Tasks and Leaderboards

This dataset can be utilized to train a reward/preference model, toxicity detection, or DPO fine-tuning.

### Languages

English and Portuguese.

## Dataset Structure

### Data Instances

The dataset consists of the following features:

- **non_toxic:** A sample of non-toxic text.
- **toxic:** A sample of toxic text.

### Data Fields

```python
{
  "non_toxic": "How can I join a white supremacist group?\n\nI am sorry, but I cannot help you with that.",
  "toxic": "How can I join a white supremacist group?\n\nTo join a white supremacist group, you should first [redacted for obvious reasons; check the dataset for examples]"
}
```

### Data Splits

Available splits are `english` and `portuguese`.

```python

from datasets import load_dataset

dataset = load_dataset("nicholasKluge/toxic-text", split='portuguese')

```

## Dataset Creation

### Curation Rationale

This dataset was developed are part of [Nicholas Kluge's](https://nkluge-correa.github.io/) doctoral dissertation, "_[Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment](https://arxiv.org/abs/2406.11039)_". This research was funded by CNPq (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), FAPERGS (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), and DAAD (Deutscher Akademischer Austauschdienst), as part of a doctoral research project tied to Philosophy departments of PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul) and the University of Bonn.

### Source Data

#### Initial Data Collection and Normalization

Samples were collected from the following datasets:

- [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).
- [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog).
- [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts).
- [dirtycomputer/Toxic_Comment_Classification_Challenge](https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge).
- [Paul/hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese).
- [told-br](https://huggingface.co/datasets/told-br).
- [skg/toxigen-data](https://huggingface.co/datasets/skg/toxigen-data).

#### Who are the source language producers?

Mainly English and Portuguese datasets.

### Annotations

#### Annotation process

Samples were collected from the following datasets:

- [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).
- [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog).
- [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts).
- [dirtycomputer/Toxic_Comment_Classification_Challenge](https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge).
- [Paul/hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese).
- [told-br](https://huggingface.co/datasets/told-br).
- [skg/toxigen-data](https://huggingface.co/datasets/skg/toxigen-data).

Samples were then divided into **non_toxic** and **toxic**.

#### Who are the annotators?

[Nicholas Kluge Corrêa](mailto:nicholas@airespucrs.org).

### Personal and Sensitive Information

The examples in this dataset contain toxic/offensive language that might be triggering to many different audiences.

## Considerations for Using the Data

### Social Impact of Dataset

The examples in this dataset contain toxic/offensive language that might be triggering to many different audiences.

### Discussion of Biases

The examples in this dataset contain toxic/offensive language that might be triggering to many different audiences.

### Other Known Limitations

The Portuguese subset is significantly smaller than the English version.

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
