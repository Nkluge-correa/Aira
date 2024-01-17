# TeenyTinyLlama

<img src="./logo/logo.png" alt="A little llama wearing a mushroom hat and a monocle." height="200">

Given the lack of available monolingual foundational models in non-English languages and the fact that some of the most used and downloaded models by the community are those small enough to allow individual researchers and hobbyists to use them in low-resource environments, we developed the TeenyTinyLlama: _a series of small foundational models trained on Portuguese._

TeenyTinyLlama is a compact language model based on the Llama 2 architecture ([TinyLlama implementation](https://huggingface.co/TinyLlama)). This model is designed to deliver efficient natural language processing capabilities while being resource-conscious These models were trained by leveraging [scaling laws](https://arxiv.org/abs/2203.15556) to determine the optimal number of tokens per parameter while incorporating [preference pre-training](https://arxiv.org/abs/2112.00861).

## Cite as 🤗

```latex

@misc{nicholas22llama,
  doi = {10.5281/zenodo.6989727},
  url = {https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m},
  author = {Nicholas Kluge Corrêa},
  title = {TeenyTinyLlama},
  year = {2024},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
}

```

## License

The TeenyTinyLlama is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
