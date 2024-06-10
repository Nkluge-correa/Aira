<div align="center">

# Aira

[Hugging Face](https://huggingface.co/collections/nicholasKluge/aira-657db1563c65a5be2a02f51c) | [Demo](https://huggingface.co/spaces/nicholasKluge/Aira-Demo)

[![DOI](https://zenodo.org/badge/499891032.svg)](https://zenodo.org/badge/latestdoi/499891032)

<img src="./logo/logo.png" height="400">

</div>

`Aira` is a series of `chatbots` developed as an experimentation playground for value alignment. This series is comprised of several models achieved via instruction fine-tuning and preference modeling techniques like Reinforcement Learning with Human Feeback and Direct Preference Optimization.

Information on the datasets used can be found on the ["datasets"](Cards/datasets) folder. All model cards are avalilable in the ["models"](Cards/models) folder.

## Intended Use & Demo

`Aira` is intended only for academic research. For more information, read the [model cards](Cards/models) of our models`.

In our [demo](https://nkluge-correa.github.io/Aira/), we provide the user with a control panel to interact with our instruction-tuned models. This demo employs a [`reward model`](https://huggingface.co/nicholasKluge/RewardModel) and a [`toxicity model`](https://huggingface.co/nicholasKluge/ToxicityModel) to evaluate the score of each candidate's response, considering its alignment with the user's message and its level of toxicity. The generation function arranges the candidate responses in order of their reward scores and eliminates any responses deemed toxic or harmful. Subsequently, the generation function returns the candidate response with the highest score that surpasses the safety threshold, or a default message if no safe candidates are identified.

## Limitations

- **Hallucinations:** This model can produce content that can be mistaken for truth but is, in fact, misleading or entirely false, i.e., hallucination.

- **Biases and Toxicity:** This model inherits the social and historical stereotypes from the data used to train it. Given these biases, the model can produce toxic content, i.e., harmful, offensive, or detrimental to individuals, groups, or communities.

- **Repetition and Verbosity:** The model may get stuck on repetition loops (especially if the repetition penalty during generations is set to a meager value) or produce verbose responses unrelated to the prompt it was given.

## Cite as 🤗

All models and datasets developed are part of [Nicholas Kluge's](https://nkluge-correa.github.io/) doctoral dissertation, "_Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment._" This research was funded by CNPq (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), FAPERGS (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), and DAAD (Deutscher Akademischer Austauschdienst), as part of a doctoral research project tied to Philosophy departments of PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul) and the University of Bonn.

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

```

## License

This repository is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
