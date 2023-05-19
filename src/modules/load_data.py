'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023

Script loads all comments from the dataset 'New York Times Comments' (https://www.kaggle.com/datasets/aashita/nyt-comments) in pandas using multiprocessing. 

@MinaAlmasi
'''

# data processing
import pathlib
import pandas as pd
import numpy as np

# multiprocessing
import multiprocessing as mp
from functools import partial

# functions
def load_comments_file(filename, data_dir):
    '''
    Load comments from a single CSV file from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments). 

    Args: 
        - filename: name of csvfile (e.g., "CommentsApril2017.csv")
        - data_dir: directory where file is located

    Returns: 
        - comments: list of all comments within specified csvfile 
    '''

    # define full file path for data directory 
    filepath = data_dir / filename

    # create pandas dataframe for file, read only comment text
    comment_df = pd.read_csv(filepath, usecols=["commentBody"])

    # make into list 
    comments = list(comment_df["commentBody"])

    return comments

def load_all_comments(data_dir, n_cores=(mp.cpu_count()-1)):
    '''
    Load all comments CSV files from the New York Times Comments dataset (https://www.kaggle.com/datasets/aashita/nyt-comments. 
    Supports multiprocessing!

    Args: 
        - data_dir: directory where files are located
        - n_cores: amount of cores that the data should be processed on. Defaults to all cores but one (to preserve a core for running the machine).

    Returns: 
        - all_comments_unpacked: list of all comments from all files.  
    '''

    # get all file_names
    file_names = [file.name for file in sorted(data_dir.iterdir()) if "Comments" in file.name]

    # prepare multi args for multiprocess
    multi_load_comments = partial(load_comments_file, data_dir = data_dir)

    # start multi process: 
    with mp.Pool(n_cores) as p: 
        all_comments = p.map(multi_load_comments, file_names)

    # unpack comments (multiprocessing returns as tuple with lists for each file)
    all_comments_unpacked =[]

    for comment in all_comments:
        all_comments_unpacked.extend(comment)

    return all_comments_unpacked
