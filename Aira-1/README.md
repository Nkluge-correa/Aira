# Aira-1

> Note: **The models in this repository have been surpassed by our Hugging Face models and the `Aira-2` series. Please [check them out!](https://huggingface.co/nicholasKluge)** 🤗

`Aira-1` is a `chatbot` designed to simulate the way a human (expert) would behave during a round of questions and answers (Q&A). `Aira-1` has many iterations, from a closed-domain chatbot based on pre-defined rules to an open-domain chatbot achieved via instruction tuning using the OpenAI API.

The creation of `Aira-1` dataset is documented in the `augmentation-factory.ipynb` file, while the `chatbot-factory.ipynb` takes the reader through the development of all iterations of our chatbot, from close-domain chatbots via `text classification` to open-domain chatbots via `conditional text generation`. In this repository, we train open-domain chatbots using the OpenAI API.

## Evaluation (Closed Domain Aira)

The `accuracy` of our text classification models is documented in the table below (for both Portuguese and English):

| Models            | Accuracy (PT) | Accuracy (EN) |
| ----------------- | ------------- | ------------- |
| Ruled-based       | 96.77%        | 95.36%        |
| Bi-LSTM           | 92.29%        | 96.98%        |
| Ensembled-Bi-LSTM | 93.73%        | 95.43%        |
| Transformer       | 97.11%        | 98.35%        |
| BERT              | **98.55%**    | **99.45%**    |

## Intended Use & Demo

`Aira-1` is intended only for academic research. To experiment with these models, you can use the `Aira-app.py` file to launch a UI based on `dash`.

## Limitations

🤥 Generative models can perpetuate the generation of pseudo-informative content, that is, false information that may appear truthful.

🤬 In certain types of tasks, generative models can produce harmful and discriminatory content inspired by historical stereotypes.
