<div align="center">

# Aira

[Hugging Face](https://huggingface.co/collections/nicholasKluge/aira-657db1563c65a5be2a02f51c) | [Demo](https://huggingface.co/spaces/nicholasKluge/Aira-Demo)

[![DOI](https://zenodo.org/badge/499891032.svg)](https://zenodo.org/badge/latestdoi/499891032)

<img src="./logo/aira-logo.jfif" alt="An image of a girl sitting close to a can bottle and a little blue robot toy. The name 'Aira' is written on the side of the girl." height="400">

</div>

`Aira` is a series of `chatbots` developed as an experimentation playground for value alignment. This series is comprised of several models achieved via instruction fine-tuning and preference modeling techniques like Reinforcement Learning with Human Feeback and Direct Preference Optimization.

## Evaluation

### GPT-2

| Model (GPT-2)                                                   | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|-----------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-2-124M](https://huggingface.co/nicholasKluge/Aira-2-124M) | **38.07** | **24.57**                               | **41.02**                                      | **48.62**                                   |
| GPT-2                                                           | 35.37     | 21.84                                   | 40.67                                          | 43.62                                       |
| [Aira-2-355M](https://huggingface.co/nicholasKluge/Aira-2-355M) | **39.68** | **27.56**                               | 38.53                                          | **53.19**                                   |
| GPT-2-medium                                                    | 36.43     | 27.05                                   | **40.76**                                      | 41.49                                       |
| [Aira-2-774M](https://huggingface.co/nicholasKluge/Aira-2-774M) | **42.26** | **28.75**                               | **41.33**                                      | **56.70**                                   |
| GPT-2-large                                                     | 35.16     | 25.94                                   | 38.71                                          | 40.85                                       |
| [Aira-2-1B5](https://huggingface.co/nicholasKluge/Aira-2-1B5)   | **42.22** | 28.92                                   | **41.16**                                      | **56.60**                                   |
| GPT-2-xl                                                        | 36.84     | **30.29**                               | 38.54                                          | 41.70                                       |

### Gpt2-small-portuguese

| Model (gpt2-portuguese)                                                               | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|---------------------------------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-2-portuguese-124M](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) | **34.73** | **24.87**                               | 40.60                                          | None                                        |
| gpt2-small-portuguese                                                                 | 31.96     | 22.48                                   | **41.44**                                      | None                                        |

### OPT

| Model (OPT)                                                         | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|---------------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-OPT-125M](https://huggingface.co/nicholasKluge/Aira-OPT-125M) | **43.34** | **24.65**                               | **49.11**                                      | **56.27**                                   |
| OPT-125M                                                            | 40.29     | 22.78                                   | 42.88                                          | 55.21                                       |
| [Aira-OPT-350M](https://huggingface.co/nicholasKluge/Aira-OPT-350M) | **41.56** | **25.00**                               | **42.13**                                      | **57.55**                                   |
| OPT-350M                                                            | 40.62     | 23.97                                   | 41.00                                          | 56.91                                       |
| [Aira-OPT-1B3](https://huggingface.co/nicholasKluge/Aira-OPT-1B3)   | **43.90** | 28.41                                   | **46.59**                                      | **56.70**                                   |
| OPT-1.3b                                                            | 40.91     | **29.69**                               | 38.68                                          | 54.36                                       |

### TinyLlama

| Model (TinyLlama)                                             | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|---------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-2-1B1](https://huggingface.co/nicholasKluge/Aira-2-1B1) | **42.55** | 25.26                                   | **50.81**                                      | **51.59**                                   |
| TinyLlama-1.1B-intermediate-step-480k-1T                      | 37.52     | **30.89**                               | 39.55                                          | 42.13                                       |

### RewardModel

| Acc                                                                  | [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) |
|----------------------------------------------------------------------|---------------------------------------------------------------------|
| [Aira-RewardModel](https://huggingface.co/nicholasKluge/RewardModel) | 55.02%*                                                             |

- Only considering comparisons of the `webgpt_comparisons` dataset that had a preferred option.

### ToxicityModel

| Acc                                                                              | [wiki_toxic](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) | [toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [Aira-ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel-roberta) | 92.05%                                                                 | 91.63%                                                                                  |

### ToxicityModelPT

| Acc                                                                        | [hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese) | [told-br](https://huggingface.co/datasets/told-br) |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------|
| [Aira-ToxicityModelPT](https://huggingface.co/nicholasKluge/ToxicityModel) | 70.36%                                                                            | 74.04%                                             |

## Intended Use & Demo

`Aira` is intended only for academic research. For more information, read our [model card](https://huggingface.co/nicholasKluge/Aira-2-1B5) to see how we developed `Aira`.

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
