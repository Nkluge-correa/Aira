# Aira - the AIRES Expert 🤓

[![DOI](https://zenodo.org/badge/499891032.svg)](https://zenodo.org/badge/latestdoi/499891032)

![chat-gif](assets/chat.gif)

[`Aira`](https://aires-playground.herokuapp.com/aira) is a `chatbot` designed to simulate the way a human (expert) would behave during a round of questions and answers (Q&A). `Aira` has many iterations, from a closed-domain chatbot based on pre-defined rules to an open-domain chatbot achieved via fine-tuning pre-trained large language models.

The creation of Aira's dataset is documented in the `augmentation-factory.ipynb` file, while the `chatbot-factory.ipynb` takes the reader through the development of all iterations of our chatbot, from close-domain chatbots via `text classification` to open-domain chatbots via `conditional text generation`. All datasets are available in the `data` folder.

Closed-domain Aira has an area of expertise that comprises topics related to AI Ethics and AI Safety research. Meanwhile, open-domain Aira can generalize for other domains. Open-domain Aira comes in four sizes, being a fine-tuned version of several sizes of GPT-3 models (made available by OpenAI).

| Models  | Size (Parameters) |
| ------- | ----------------- |
| Ada     | 350M              |
| Babbage | 3B                |
| Curie   | 13B               |
| Davinci | 175B              |

## Metrics

## Closed Domain Aira

The `accuracy` of our text classification models is documented in the table below (for both Portuguese and English):

| Models            | Accuracy (PT) | Accuracy (EN) |
| ----------------- | ------------- | ------------- |
| Ruled-based       | 96.77%        | 95.36%        |
| Bi-LSTM           | 92.29%        | 96.98%        |
| Ensembled-Bi-LSTM | 93.73%        | 95.43%        |
| Transformer       | 97.11%        | 98.35%        |
| BERT              | **98.55%**    | **99.45%**    |

## Limitations

Our open-domain conversational chatbots were achieved via conditional text generation. This approach has a lot of limitations. Even though we can make a chatbot that can answer questions about anything, forcing the model to produce good-quality responses is hard. And by good, we mean factual and nontoxic responses. This leads us to two of the most common problems of generative models used in conversational applications:

- 🤥 Generative models can perpetuate the generation of pseudo-informative content, that is, false information that may appear truthful. For example, multi-modal generative models can be used to create images with untruthful content, while language models for text generation can automate the generation of misinformation.

- 🤬 In certain types of tasks, generative models can generate toxic and discriminatory content inspired by historical stereotypes against sensitive attributes (for example, gender, race, and religion). Unfiltered public datasets may also contain inappropriate content, such as pornography, racist images, and social stereotypes, which can contribute to unethical biases in generative models. Furthermore, when prompted with non-English languages, some generative models may perform poorly.

## Demo

In our [demo](https://aires-playground.herokuapp.com/aira), we provide the user with a control panel to interact with our open domain models. The closed domain models are all available in this repository for download. We also provide the possibility to users to conduct conversations with a prompt-tune version of [ChatGPT](https://geo-not-available.blenderbot.ai/) (`Chat-Aira`), so that people can compare the capabilities of open-domain conversational chatbots (`Aira`) created via fine-tunning and large language models created via RLHF (`Chat-Aira`).

We made available a copy of our demo application (built using [`dash`](https://dash.plotly.com/)) in this repository. Assets can be found in the `assets` folder and the app in the `Aira-app.py` file.

If you have any questions concerning `Aira`, please contact [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org).

If you are not satisfied with `Aira's` performance or would like to help us improve the capabilities of our system, or if you would like to complain about any type of message produced by `Aira`, please contact [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org).

## Cite as 🤗

---

```latex

@misc{nicholas22aira,
  doi = {10.5281/zenodo.6989727},
  url = {https://github.com/Nkluge-correa/Aira-EXPERT},
  author = {Nicholas Kluge Corrêa and Carolina Del Pino},
  title = {Aira - the AIRES Expert},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  note = {Last updated on 16 April 2023},
}

```

Aira was developed by [Nicholas Kluge](https://nkluge-correa.github.io/) and [Carolina Del Pino](http://lattes.cnpq.br/6291330432531578).
