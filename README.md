# Simple RNN Language Model

Quite simple RNN using LSTMs as its main units and words as tokens. <br>The model has been trained on two <b>Jack London</b>'s novels (open source, Project Gutenberg):
* 360 epochs (with the increasing size of batches);
* 650KB of text (~93k sentences);
* 10k words in vocabulary;
* a 5-word prompt;
* generating a 200-word text.

## Getting Started
Requirements: TensorFlow 2.12.1+ and Python 3.9.17+.

1. Create an instance of the language model based on your choice of RNN architecture.

```python
# Example for creating an instance of WordRNN
language_model = WordRNN(dictionary_size=10_000, sentence_length=5)
language_model.compile_model()
```

2. Load the trained model to generate text.
```python
# Generating text for unknown prompts
language_model.generate('I scarcely know where to begin... ', temperature=1.5)
```

## Examples

Explore the capabilities of WordRNN with our provided examples:

- Test the model's accuracy when given a known prompt (it will reproduce the chosen part of one of the novels).
- Sample text for known prompts (it will combine sentences and phrases from the novels).
- Generate text for unknown prompts to witness some creative language generation (it will generate completely
  new phrases or combine the learnt ones to create a new sentence).

<p align="left">
    <img width="100%" src="https://github.com/TairYerniyazov/SimpleLanguageModel/blob/main/example.gif" 
      alt="Structure of the model (layers)">
</p>

While the text may not convey any meaningful message, it is evident that the model has grasped the fundamentals of 
forming coherent phrases, maintaining the correct grammatical structure and word relationships in some parts of
the sentences. For instance, it appropriately combines articles with nouns and matches pronouns with verbs. 
However, it is crucial to note that the dataset used for training is relatively small. 
Furthermore, it's worth mentioning that when it comes to comprehending the basic connections between 
individual words in a sentence over extended distances, LSTM units fall short compared to transformers. 

## Architecture

Use the `show_structure` method to visualise the architecture of the model, giving you insights into its layers and structure.

<p align="center">
    <img width="50%" src="https://github.com/TairYerniyazov/SimpleLanguageModel/blob/main/structure/model_structure.png" 
      alt="Structure of the model (layers)">
</p>
