'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023

Script contains functions for processing the comments from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments) prior to training an LSTM on the data.
This processing entails cleaning text, tokenizing data, creating input sequences and padding those input sequences. 

Some functions were fully or partially developed in class. This is denoted beside the function in a comment. 

@MinaAlmasi
'''

# data processing
import string, re, os 
import pandas as pd
import numpy as np

# tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku 

# add custom module for logging 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from utils.custom_logging import custom_logger

# functions
def clean_comment(comment): # part of code for this function has been adapted from session_08 inclass.
    '''
    Clean comment from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments).

    Args: 
        - comment: unprocessed comment 
    
    Returns: 
        - comment: cleaned comment
    '''

    # remove html breaks <br/>
    comment = re.sub("<br/>", " ", comment)

    # remove puncutation and lower case the text
    comment = "".join(word for word in comment if word not in string.punctuation).lower()

    # ensure comment is in the right encoding
    comment = comment.encode("utf8").decode("ascii",'ignore')
    
    return comment


def tokenize(data):
    '''
    Tokenize data using tensorflow's Tokenizer(). 

    Args: 
        - data: data to be tokenized (should be cleaned !)

    Returns: 
        - tokenizer: fitted tokenizer
        - vocabulary_size: number of all unique tokens in data (+ a <unk> token to denote unknown tokens in vocabulary)
    '''

    # initialize tokenizer
    tokenizer = Tokenizer()

    # fit tokenizer 
    tokenizer.fit_on_texts(data)

    # define vocabulary
    vocabulary_size = len(tokenizer.word_index) + 1 # +1 accounts for <unk> token for words not present in vocabulary

    return tokenizer, vocabulary_size


def get_sequence_of_tokens(tokenizer, data): # this function was developed in class
    '''
    Convert data to tokens made into different sequences to serve as input when training model

    Args: 
        - tokenizer: tokenizer that has already been fitted on the data
        - data: data (same that the tokenizer has been fitted on)

    Returns:
        - input_sequences: N-gram sequences of tokens
    '''

    input_sequences = []

    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    return input_sequences


def generate_padded_sequences(input_sequences, vocabulary_size): # this function was developed in class, but modified slightly
    '''
    Add padding to input sequences, so that all input sequences are of same length 
    (by adding 0s to the beginning of all input sequences so all input sequences match the length of the longest sequence)

    Args: 
        - input_sequences: N-gram sequences of tokens 
        - vocabulary_size: Number of all unique tokens in data (+ a <unk> token to denote unknown tokens in vocabulary)

    Returns: 
        - predictors: input vectors (X) used for training purposes (N-gram sequences)
        - label: label vectors (Y) used for training purposes (next word of the predictors/N-gram sequences)
        - max_sequence_len: the length of the longest sequence
    '''

    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    
    label = ku.to_categorical(label, 
                            num_classes=vocabulary_size)
    
    return predictors, label, max_sequence_len


def get_sequences_with_pad(tokenizer, vocabulary_size, data):
    '''
    Get input sequences with padding from tokenizer and vocabulary_size (amount of tokens). 

    Args: 
        - tokenizer: tokenizer already fitted on the data
        - vocabulary_size: number of all unique tokens in data (+ a <unk> token to denote unknown tokens in vocabulary)
        - data: preprocessed (clean) data 

    Returns: 
        - predictors: input vectors (X) used for training purposes (N-gram sequences)
        - label: label vectors (Y) used for training purposes (next word of the predictors/N-gram sequences)
        - max_sequence_len: the length of the longest sequence
    '''

    # initialize logger 
    logging = custom_logger("get-pad-sequences")

    # create input sequences
    logging.info("Getting sequences")
    input_sequences = get_sequence_of_tokens(tokenizer, data)

    # pad input sequences to make them all same length 
    logging.info("Padding sequences")
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, vocabulary_size)

    return predictors, label, max_sequence_len

def process_comments(data):
    '''
    Pipeline to process comments from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments) prior to training a LSTM model for text generation. 
    The pipeline will clean raw comments, 

    Args: 
        - data: list of comments, unprocessed
    
    Returns: 
        - tokenizer: fitted tokenizer
        - max_sequence_len: the length of the longest sequence
        - vocabulary_size: number of all unique tokens in data (+ a <unk> token to denote unknown tokens in vocabulary)
        - predictors: input vectors (X) used for training purposes (N-gram sequences)
        - label: label vectors (Y) used for training purposes (next word of the predictors/N-gram sequences)
    '''

    # intialize logger
    logging = custom_logger("process_comments")
    
    # clean comments
    logging.info("CLEANING")
    data_clean = [clean_comment(comment) for comment in data]

    # tokenize comments
    logging.info("TOKENIZING")
    tokenizer, vocabulary_size = tokenize(data_clean)

    # create input sequences and pad them 
    logging.info("CREATING INPUT SEQUENCES")
    predictors, label, max_sequence_len = get_sequences_with_pad(tokenizer, vocabulary_size, data_clean)

    return tokenizer, max_sequence_len, vocabulary_size, predictors, label

