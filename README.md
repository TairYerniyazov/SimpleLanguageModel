# WordRNN: Recurrent Neural Network Language Model

Quite simple RNN using LSTMs as its main units and words as tokens. The model has been trained on several Jack London's novels
(60 epochs, 1.4Mb of text, 10k words in vocabulary).

## Getting Started

1. Create an instance of the language model based on your choice of RNN architecture.

```python
# Example for creating an instance of WordRNN
language_model = WordRNN(dictionary_size=10_000, sentence_length=10)
language_model.compile_model()
```

2. Load the trained model to generate text.
```python
# Generating text for unknown prompts
language_model.generate('I scarcely know where to begin... ', temperature=1.5)
```

## Examples

Explore the capabilities of WordRNN with our provided examples:

- Test the model's accuracy when given a known prompt.
- Sample text for known prompts (prompts that were in the training datasets but may not produce the same continuation).
- Generate text for unknown prompts to witness the model's creative language generation.

<p align="left">
    <img width="100%" src="https://github.com/TairYerniyazov/SimpleLanguageModel/blob/main/example.gif" 
      alt="Structure of the model (layers)">
</p>

## Architecture

Use the `show_structure` method to visualize the architecture of your WordRNN model, giving you insights into its layers and structure.

<p align="center">
    <img width="50%" src="https://github.com/TairYerniyazov/SimpleLanguageModel/blob/main/structure/model_structure.png" 
      alt="Structure of the model (layers)">
</p>
