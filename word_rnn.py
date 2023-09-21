# Disabling all TensorFlow debugging logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Uncomment if you want to use CPUs only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class WordRNN(tf.keras.Model):

    def __init__(self, dictionary_size, sentence_length):
        super(WordRNN, self).__init__()
        
        self.sentence_length = sentence_length
        self.dictionary_size = dictionary_size
        self.output_dictionary = []

        # Auxiliary layer
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=dictionary_size,
            pad_to_max_tokens=True,
            standardize=None,
            output_mode='int',
            output_sequence_length=sentence_length)
        
        # Main layers
        self.embedding = tf.keras.layers.Embedding(dictionary_size + 1, output_dim=32, 
                                                   input_length=sentence_length)
        self.lstm_1 = tf.keras.layers.LSTM(128, activation='relu', recurrent_dropout=0.4,
                                           return_sequences=True)
        self.batch_normalizer_1 = tf.keras.layers.BatchNormalization(synchronized=True)
        self.dropout_layer = tf.keras.layers.Dropout(.3)
        self.lstm_2 = tf.keras.layers.LSTM(64, activation='relu', recurrent_dropout=0.3)
        self.batch_normalizer_2 = tf.keras.layers.BatchNormalization(synchronized=True)
        self.dense =  tf.keras.layers.Dense(dictionary_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.batch_normalizer_1(self.lstm_1(x))
        x = self.dropout_layer(x)
        x = self.batch_normalizer_2(self.lstm_2(x))
        return self.dense(x)
    
    def compile_model(self):
        self(tf.keras.Input(shape=(self.sentence_length,)))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
    
    def train(self, datasets_paths, n_epochs):
        x_train, y_train = self._extract_samples(datasets_paths)
        history = self.fit(x_train, y_train, batch_size=128, epochs=n_epochs, validation_split=0.2)
        print(">>> Model training has finished.")

    def _extract_samples(self, file_paths):
        """ Method preparing input sentences and target words for training """
        
        # Extracting sample sentences and words
        text = ''
        for file_path in file_paths:
            with open(file_path) as file:
                text += file.read() + ' '
        text = text.split()
        x, y = [], []
        for word in range(0, len(text) - self.sentence_length, 1):
            x.append(text[word: word + self.sentence_length])
            y.append(text[word + self.sentence_length])
        x = [[' '.join(word)] for word in x]
        y = [[''.join(word)] for word in y]

        # Adapting the TextVectorization layer to the dataset
        dataset = tf.data.Dataset.from_tensor_slices(text)
        self.vectorize_layer.adapt(dataset.batch(128))
        
        # Preparing tokens (tensors)
        vectorizer = self._get_vectorizer()
        sentences = np.array(vectorizer.predict(x))
        y = np.array(vectorizer.predict(y))
        y = np.delete(y, np.s_[1:self.sentence_length], 1)
        target_words = np.zeros((len(y), self.dictionary_size))
        for i in range(len(y)):
            target_words[i][y[i]] = 1
        self.output_dictionary = self.vectorize_layer.get_vocabulary()
        
        # Serialising and saving the adapted vocabulary for TextVectorization layer
        with open(file_path + 'vocabulary.pkl', 'wb') as out_file:
            pickle.dump(self.output_dictionary, out_file)
        
        return sentences, target_words

    def _get_vectorizer(self):
        vectorizer = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            self.vectorize_layer
        ])
        return vectorizer
    
    def save(self, file_path):
        self.save_weights(file_path + 'model.tf', save_format='tf')
        print(">>> The weights of the model have been succesfully saved: " + file_path)

    def load(self, file_path):
        # Deserializing and uploading the adapted vocabulary for TextVectorization layer
        with open(file_path + 'vocabulary.pkl', 'rb') as in_file:
            self.output_dictionary = pickle.load(in_file)
        self.vectorize_layer.set_vocabulary(self.output_dictionary)
        # Loading the weights of the model
        self.load_weights(file_path + 'model.tf').expect_partial()
        print("\n>>> Loading the weights of the from file " + file_path 
              + " model has been completed.")

    def generate(self, prompt, temperature=1.0, original_text=False):
        sys.stdout.write("\n\n>>> temperature=" + str(temperature) + " \n\n")
        # Preprocessing the initial input
        sys.stdout.write('Prompt: ' + prompt + '\n\n')
        vectorizer = self._get_vectorizer()
        prompt = vectorizer.predict([[prompt]], verbose=0)
        prompt = np.array(prompt)
        for i in range(self.sentence_length):
            if prompt[0][i] == 0:
                prompt[0] = np.roll(prompt, self.sentence_length - i)
        
        # Generating next words
        for i in range(200):
            output = self.predict(prompt, verbose=0)
            if original_text:
                output = np.argmax(output)
                sys.stdout.write(self.output_dictionary[output] + " ")
            else:
                output = [tf.math.log(output[0]) / temperature]
                output = tf.random.categorical(output, num_samples=1)
                if output > len(self.output_dictionary):
                    output = np.array([[0]])
                sys.stdout.write(self.output_dictionary[output[0][0]] + " ")

            sys.stdout.flush()
            prompt[0] = np.roll(prompt, -1)
            prompt[0][9] = output
        sys.stdout.write('\n\n')

    def show_structure(self, file_path):
        """ Method generating the visualised structure of the model (layers).
            Use it if you have "pydot" and "graphviz" installed."""
        inputs = tf.keras.Input(shape=(self.sentence_length,))
        model_func = tf.keras.Model(inputs, self.call(inputs))
        tf.keras.utils.plot_model(model_func, to_file=file_path, show_shapes=True)


# Creating an instance of the language model based on the LSTM 
# recurrent neural network using individual words as tokens 
language_model = WordRNN(10_000, 10)
language_model.compile_model()

# Loading the trained model
language_model.load('saved_model/')

""" ********* UNCOMMENT TO TRAIN AND SAVE THE MODEL: **********

# Training the model on the chosen datasets
language_model.train([
    'datasets/white_fang_by_jack_london.txt',
    'datasets/the_call_of_the_wild_by_jack_london.txt',
    'datasets/martin_eden_by_jack_london.txt'
    ], n_epochs=20)

# Saving the trained model
language_model.save('saved_model/')

# Saving the visualisation of the model structure
language_model.show_structure('structure/model_structure.png')

*************************************************************** """

# # Testing the accuracy of the model when giving it the known prompt (model capacity,
# # the prompts have been observed, so we expect to see the original continuation)
language_model.generate('In the morning it was Henry who awoke first and ', original_text=True)

# # Sampling for known prompts (were in the datasets, but next words do not need to be the same)
language_model.generate('In the morning it was Henry who awoke first and ', temperature=1.5)

# Generating text for unknown prompts
language_model.generate('I scarcely know where to begin... ', temperature=1.5)
