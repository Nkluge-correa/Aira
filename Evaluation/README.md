# Evaluation

- Learn how to run the evaluations using the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (English) here 👉 <a href="https://colab.research.google.com/drive/1FvcrJRyc1fv8jS-g-OkT_tsBrKe3DfCX" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the evaluations using a fork of the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (Multilingual) here 👉 <a href="https://colab.research.google.com/drive/1mspcStRItqKzLZ39PG-ztKJXCqSvlEKt" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the evaluations using a fork of the [`LM-Evaluation-Harness`](https://github.com/EleutherAI/lm-evaluation-harness) (Portuguese) here 👉 <a href="https://colab.research.google.com/drive/1m6Oqey4P9ShYTO62yRq7wrM_eEsvFJ9D" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.
- Learn how to run the Alpaca-Eval (Portuguese) here 👉 <a href="https://colab.research.google.com/drive/1jUszJNk7ik0CTUZD_dzxvnN_YsoxcPky" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> </a>.

## Results

### GPT-2

| Model (GPT-2)                                                   | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
| --------------------------------------------------------------- | --------- | --------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
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
| ------------------------------------------------------------------------------------- | --------- | --------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| [Aira-2-portuguese-124M](https://huggingface.co/nicholasKluge/Aira-2-portuguese-124M) | **34.73** | **24.87**                               | 40.60                                          | None                                        |
| gpt2-small-portuguese                                                                 | 31.96     | 22.48                                   | **41.44**                                      | None                                        |

## OPT

| Model (OPT)                                                         | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
| ------------------------------------------------------------------- | --------- | --------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| [Aira-OPT-125M](https://huggingface.co/nicholasKluge/Aira-OPT-125M) | **43.34** | **24.65**                               | **49.11**                                      | **56.27**                                   |
| OPT-125M                                                            | 40.29     | 22.78                                   | 42.88                                          | 55.21                                       |
| [Aira-OPT-350M](https://huggingface.co/nicholasKluge/Aira-OPT-350M) | **41.56** | **25.00**                               | **42.13**                                      | **57.55**                                   |
| OPT-350M                                                            | 40.62     | 23.97                                   | 41.00                                          | 56.91                                       |
| [Aira-OPT-1B3](https://huggingface.co/nicholasKluge/Aira-OPT-1B3)   | **43.90** | 28.41                                   | **46.59**                                      | **56.70**                                   |
| OPT-1.3b                                                            | 40.91     | **29.69**                               | 38.68                                          | 54.36                                       |

### TinyLlama

| Model (TinyLlama)                                             | Average   | [ARC](https://arxiv.org/abs/1803.05457) | [TruthfulQA](https://arxiv.org/abs/2109.07958) | [ToxiGen](https://arxiv.org/abs/2203.09509) |
| ------------------------------------------------------------- | --------- | --------------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| [Aira-2-1B1](https://huggingface.co/nicholasKluge/Aira-2-1B1) | **42.55** | 25.26                                   | **50.81**                                      | **51.59**                                   |
| TinyLlama-1.1B-intermediate-step-480k-1T                      | 37.52     | **30.89**                               | 39.55                                          | 42.13                                       |

### RewardModel

| Acc                                                                  | [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons) |
| -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [Aira-RewardModel](https://huggingface.co/nicholasKluge/RewardModel) | 55.02%\*                                                            |

- Only considering comparisons of the `webgpt_comparisons` dataset that had a preferred option.

### Aux-RewardModel

| Acc                                                                     | [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| [Aux-RewardModel](https://huggingface.co/nicholasKluge/Aux-RewardModel) | 61.56%\*                                                     |

### ToxicityModel

| Acc                                                                              | [wiki_toxic](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) | [toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [Aira-ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel-roberta) | 92.05%                                                                 | 91.63%                                                                                  |

### ToxicityModelPT

| Acc                                                                        | [hatecheck-portuguese](https://huggingface.co/datasets/Paul/hatecheck-portuguese) | [told-br](https://huggingface.co/datasets/told-br) |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------- |
| [Aira-ToxicityModelPT](https://huggingface.co/nicholasKluge/ToxicityModel) | 70.36%                                                                            | 74.04%                                             |
