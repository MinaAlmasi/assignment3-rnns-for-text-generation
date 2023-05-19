'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023. 

Script contains functions for creating an LSTM, generating text with a fitted model and saving model card + loss curve. 
The first two were developed in class, although modified slightly.

@MinaAlmasi
'''

# import model 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# system
import pathlib

# other tools
import numpy as np
import matplotlib.pyplot as plt

# functions for creating model and generating text from trained model
def create_model(max_sequence_len:int, embedding_layer_size: int, hidden_layer_size:int, vocabulary_size:int): # developed in class, but slightly modified
    '''
    Create basic LSTM model

    Args: 
        - max_sequence_len: highest possible sequence length 
        - embedding_layer_size: size of the embedding layer
        - hidden_layer_size: size of the hidden layer
        - vocabulary_size: total words that model knows (tokens)

    Returns:
        - model: tensorflow LSTM model
    '''

    # define input length
    input_len = max_sequence_len - 1

    # intialize model
    model = Sequential()
    
    # add input embedding layer
    model.add(Embedding(vocabulary_size, 
                        embedding_layer_size, 
                        input_length=input_len))
    
    # add hidden layer 1 (LSTM layer)
    model.add(LSTM(hidden_layer_size))
    model.add(Dropout(0.1))
    
    # add output layer
    model.add(Dense(vocabulary_size, 
                    activation='softmax'))

    # compile loss 
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len): # modified slightly from class: tokenizer param added, commented
    '''
    Generate a continuation from a text input (seed_text). 
    E.g. "This article is .." -> "This article is the not factual and should be illegal." 

    Args: 
        - seed_text: input text to generate output text from 
        - next_words: n words to generate from the seed_text
        - model: fitted tf model
        - tokenizer: fitted tokenizer
        - max_sequence_len: highest possible sequence length

    Returns:
        - seed_text: original seed_text + continuation (new output words added)
    '''

    for _ in range(next_words): # next-word prediction happening in for loop i.e., make this loop continue for amount next words defined
        # make seed_text into sequence of tokens
        token_list = tokenizer.texts_to_sequences([seed_text])[0] 

        # pad seed_text sequence
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        
        # make model predict word from seed_text
        predicted = np.argmax(model.predict(token_list), 
                                            axis=1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items(): 
             # extract predicted (next) word in tokenizer word_index 
            if index == predicted:
                output_word = word
                break
        
        # add predicted (next) word as a continuation of seed_text
        seed_text += " "+output_word 

    return seed_text

# functions for model info! 

def save_model_card(model, n_epochs:int, n_data:int, max_sequence_len:int, savepath:str):
    '''
    Save model card (summary of layers, trainable parameters) as txt file in desired directory (savepath).

    Args: 
        - model: model with defined layers
        - n_epochs: amount of epochs the model has been trained for
        - n_data: number of data rows the model has been trained on
        - max_sequence_len: highest possible sequence length
        - savepath: path where model card should be saved e.g. models/file.txt
    
    Outputs: 
        - .txt file with model summary
    '''

    # define full path
    filepath = savepath / f"card_model_{max_sequence_len}.txt"
    
    # write model summary as txt
    with open(filepath,'w') as file:
        file.write(f"Model with a max sequence length of {max_sequence_len}, trained on {n_data} rows of data \n") 
        model.summary(print_fn=lambda x: file.write(x + '\n'))

def plt_training_loss(history, model_name, savepath:str):
    '''
    Save a plot of model training loss to desired directory (savepath).

    Args: 
        - history: history of trained model
        - savepath: path where the plot should be saved e.g., models/plot.png

    Outputs: 
        - .png file of loss curve
    '''

    # plot loss
    plt.plot(history.history["loss"])

    # set title, ylabel, xlabel and legend
    plt.title("Training Loss by Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    # save plot
    plt.savefig(savepath / f"loss_{model_name}.png")