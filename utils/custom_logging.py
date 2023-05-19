'''
Utils script for Assignment 3, Language Analytics, Cultural Data Science, F2023

NB. This script is taken from past projects in Language Analytics and Visual Analytics. 

The following script contains a single function that customises a logging logger to display messages in the terminal. 

@MinaAlmasi
'''


import logging

def custom_logger(name):
    '''
    Custom logger for displaying messages to console while running scripts.

    Args: 
        - name: name of logger
    Returns: 
        - logger object to be used in functions and scripts

    '''

    # define loggger
    logger = logging.getLogger(name)

    # set level of logging (level of detail of what should be logged)
    logger.setLevel(level=logging.INFO)

    # instantiate console logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler) # add handler to overall logger

    # define formatting of logging
    formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    console_handler.setFormatter(formatter) # add formatting to console handler

    return logger
