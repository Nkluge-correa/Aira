# Aira

[![DOI](https://zenodo.org/badge/499891032.svg)](https://zenodo.org/badge/latestdoi/499891032)

<img src="./logo/aira-logo.jfif" alt="An image of a girl sitting close to a can bottle and a little blue robot toy. The name 'Aira' is written on the side of the girl." height="400">

[`Aira`](https://nkluge-correa.github.io/Aira/) is a series of `chatbots` developed as an experimentation playground for value alignment. While series 1 ([`Aira-1`](https://github.com/Nkluge-correa/Aira/tree/master/Aira-1)) represents our first attempts to develop conversational bots via text classification and conditional text generation, series 2 ([`Aira-2`](https://github.com/Nkluge-correa/Aira/tree/master/Aira-2)) is comprised of several models achieved via instruction finetuning and preference modeling techniques like Reinforcement Learning with Human Feeback and Direct Preference Optimization.

All models relating to the `Aira-1` series can be downloaded via the URLs in [this repo](https://github.com/Nkluge-correa/Aira/tree/master/Aira-1/aira) while the entire `Aira-2` series is available on the [Hugging Face hub](https://huggingface.co/nicholasKluge).

This repository also contains the source code for the pre-training of several types of language models (`Bert`, `GPT-2`, `Llama`), using common machine learning frameworks and libraries, like `transformers`, `datasets`, and `torch`. The source code of our first attempt to train such models is documented in the `old-pre-training-scripts` folder (models also available on the [Hugging Face hub](https://huggingface.co/AiresPucrs/bert-base-wikitext)), while the `TeenyTinyLlama` folder contains the source code for training a small version of the Llama-2 architecture with a Portuguese corpus, following the [Chinchila scaling laws](https://arxiv.org/abs/2203.15556) while also incorporating [preference pre-training](https://arxiv.org/abs/2112.00861).

All models and datasets developed are part of [Nicholas Kluge's](https://nkluge-correa.github.io/) doctoral dissertation, "_Dynamic Normativity: Necessary and Sufficient Conditions for Value Alignment._" This research was funded by CNPq (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), FAPERGS (Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul), and DAAD (Deutscher Akademischer Austauschdienst), as part of a doctoral research project tied to Philosophy departments of PUCRS (Pontifícia Universidade Católica do Rio Grande do Sul) and the University of Bonn.

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

```
