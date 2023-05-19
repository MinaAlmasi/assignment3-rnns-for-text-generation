'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023

This script contains a pipeline to train LSTM on the comments from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments). 
The pipeline will both load the data, process that data (tokenization, padding input sequences), train the model and finally save the model. 

    To run the script in the terminal, type: 
        python src/train_model.py 
    
    The additional arguments are (if left unspecified, defaults will run)
        -n: how many comments the fitting should run on (n_samples). Defaults to all of them. May be too computationally heavy!
        -hl: size of hidden layer. Defaults to 30. 
        -el: size of embedding layer. Defaults to 10.
        -e: how many epochs the model training should be. Defaults to 50.

    For instance: 
        python src/train_model.py -n 1000 -hl 40 

@MinaAlmasi
'''

# data processing 
import numpy as np
import argparse

# model
import tensorflow as tf

# system tools
import pathlib
import multiprocessing as mp
import sys
import pickle

# custom modules for loading and processing 
from modules.load_data import load_all_comments
from modules.process_data import process_comments

# custom modules for train pipeline 
from modules.model_fns import create_model, save_model_card, plt_training_loss

# custom logger 
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-n", "--n_samples", help = "amount of sample data that you want to use", type = int, default=None)
    parser.add_argument("-el", "--embedding_layer", help = "size of embedding layer", type = int, default=10)
    parser.add_argument("-hl", "--hidden_layer", help = "size of first and only hidden layer", type = int, default=30)
    parser.add_argument("-e", "--epochs", help = "amount epochs you want to run the model for", type = int, default=50)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args


def main():
    # set seed 
    tf.random.set_seed(129)
    np.random.seed(129)

    # define filepath, datapath
    path = pathlib.Path(__file__) 
    datapath = path.parents[1] / "data"

    # initialize logger 
    logging = custom_logger("pipeline-logger")

    # initialize args
    args = input_parse()

    # get all cores -1 for loading datafiles parallel
    n_cores = mp.cpu_count() - 1

    logging.info("LOADING FILES") 
    all_comments = load_all_comments(datapath, n_cores)

    # subset 
    if args.n_samples != None: # if args.n_samples is not None, then subset data:
        all_comments = all_comments[:args.n_samples]
    
    # user msg on how many comments is processes
    logging.info(f"Running pipeline with {len(all_comments)} comments")
    
    # return tokenized data 
    tokenizer, max_sequence_len, vocabulary_size, predictors, label = process_comments(all_comments)

    # create model 
    model = create_model(max_sequence_len=max_sequence_len, embedding_layer_size=args.embedding_layer, hidden_layer_size=args.hidden_layer, vocabulary_size=vocabulary_size)

    # fit model 
    logging.info("FITTING MODEL") 
    history = model.fit(predictors, label, epochs = args.epochs, batch_size=128, verbose=1)

    # make folder for model contents 
    model_folder = path.parents[1] / "models" / f"model_{max_sequence_len}"  # define folder
    model_folder.mkdir(parents=True, exist_ok=True) # make if it does not exist

    # save model 
    model.save(model_folder / f"model_{max_sequence_len}.h5")

    # save tokenizer
    tokenizer_path = model_folder / f"tokenizer_{max_sequence_len}.pickle"

    with open(tokenizer_path, "wb") as handle: 
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save model card
    save_model_card(model, args.epochs, len(all_comments), max_sequence_len, model_folder)

    # save loss curve
    plt_training_loss(history, f"model_{max_sequence_len}", model_folder)
    logging.info("FITTING COMPLETED. SAVED MODEL AND INFO")

if __name__ == "__main__":
    main()

