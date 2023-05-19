'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023

Script contains functions for saving and loading a tokenizer

@MinaAlmasi
'''

import pickle

def save_tokenizer(tokenizer, tokenizerpath):
    '''
    Save fitted tokenizer.

    Args: 
        - tokenizer: fitted tokenizer object
        - tokenizerpath: path where the tokenizer should be saved e.g., models/tokenizer.pickle
    '''

    with open(tokenizerpath, "wb") as handle: 
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(tokenizerpath): 
    '''
    Load saved tokenizer.

    Args: 
        - tokenizerpath: where the tokenizer is located e.g., models/tokenizer.pickle
    '''

    with open(tokenizerpath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return tokenizer