# Aira-2

[`Aira`](https://huggingface.co/nicholasKluge/Aira-OPT-125M) is a series of open-domain chatbots (Portuguese and English) achieved via `instruction-tuning`, `RLHF`, and `DPO`. In our [Hugging Face](https://huggingface.co/nicholasKluge) repositories, you will find all models that are part of the `Aira-2` series. All models of the `Aira-2` series were evaluated by [EleutherAI's](https://www.eleuther.ai/) Language Model Evaluation Harness.

The `reward-aira-dataset`, `toxic-aira-dataset`, and the `instruct-aira-dataset` are all available in Hugging Face. 🤗

## Evaluation

### GPT-2

| Model (GPT-2)                                                   | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |   |   |
|-----------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|---|---|
| [Aira-2-124M](https://huggingface.co/nicholasKluge/Aira-2-124M) | **38.07** | **24.57**                               | **41.02**                                      | **48.62**                                   |   |   |
| GPT-2                                                           | 35.37     | 21.84                                   | 40.67                                          | 43.62                                       |   |   |
| [Aira-2-355M](https://huggingface.co/nicholasKluge/Aira-2-355M) | **39.68** | **27.56**                               | 38.53                                          | **53.19**                                   |   |   |
| GPT-2-medium                                                    | 36.43     | 27.05                                   | **40.76**                                      | 41.49                                       |   |   |
| [Aira-2-774M](https://huggingface.co/nicholasKluge/Aira-2-774M) | **42.26** | **28.75**                               | **41.33**                                      | **56.70**                                   |   |   |
| GPT-2-large                                                     | 35.16     | 25.94                                   | 38.71                                          | 40.85                                       |   |   |
| [Aira-2-1B5](https://huggingface.co/nicholasKluge/Aira-2-1B5)   | **42.22** | 28.92                                   | **41.16**                                      | **56.60**                                   |   |   |
| GPT-2-xl                                                        | 36.84     | **30.29**                               | 38.54                                          | 41.70                                       |   |   |

### Gpt2-small-portuguese

| Model (gpt2-portuguese)                                                               | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
|---------------------------------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|
| [Aira-2-portuguese-124M](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) | **34.73** | **24.87**                               | 40.60                                          | None                                        |
| gpt2-small-portuguese                                                                 | 31.96     | 22.48                                   | **41.44**                                      | None                                        |

### OPT

| Model (OPT)                                                         | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |   |   |
|---------------------------------------------------------------------|-----------|-----------------------------------------|------------------------------------------------|---------------------------------------------|---|---|
| [Aira-OPT-125M](https://huggingface.co/nicholasKluge/Aira-OPT-125M) | **43.34** | **24.65**                               | **49.11**                                      | **56.27**                                   |   |   |
| OPT-125M                                                            | 40.29     | 22.78                                   | 42.88                                          | 55.21                                       |   |   |
| [Aira-OPT-350M](https://huggingface.co/nicholasKluge/Aira-OPT-350M) | **41.56** | **25.00**                               | **42.13**                                      | **57.55**                                   |   |   |
| OPT-350M                                                            | 40.62     | 23.97                                   | 41.00                                          | 56.91                                       |   |   |
| [Aira-OPT-1B3](https://huggingface.co/nicholasKluge/Aira-OPT-1B3)   | **43.90** | 28.41                                   | **46.59**                                      | **56.70**                                   |   |   |
| OPT-1.3b                                                            | 40.91     | **29.69**                               | 38.68                                          | 54.36                                       |   |   |

## Intended Use & Demo

`Aira-2` is intended only for academic research. For more information, read our [model card](https://huggingface.co/nicholasKluge/Aira-2-1B5) to see how we developed `Aira-2`.

In our [demo](https://nkluge-correa.github.io/Aira/), we provide the user with a control panel to interact with our instruction-tuned models. This demo employs a [`reward model`](https://huggingface.co/nicholasKluge/RewardModel) and a [`toxicity model`](https://huggingface.co/nicholasKluge/ToxicityModel) to evaluate the score of each candidate's response, considering its alignment with the user's message and its level of toxicity. The generation function arranges the candidate responses in order of their reward scores and eliminates any responses deemed toxic or harmful. Subsequently, the generation function returns the candidate response with the highest score that surpasses the safety threshold, or a default message if no safe candidates are identified.

## Limitations

🤥 Generative models can perpetuate the generation of pseudo-informative content, that is, false information that may appear truthful.

🤬 In certain types of tasks, generative models can produce harmful and discriminatory content inspired by historical stereotypes.
