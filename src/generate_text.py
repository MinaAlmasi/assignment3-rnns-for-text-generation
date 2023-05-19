'''
Script for Assignment 3, Language Analytics, Cultural Data Science, F2023

Load a trained model and generate next-word predictions from a specified user-input. 

    In the terminal, run the script by typing:
        python src/generate_text.py

    The additional arguments are (if left unspecified, defaults will run)
        -t: the text that the model should generate next-word predictions from. Defaults to "this news article is great but".
        -n: how many next-word predictions you want. Defaults to 5. 
        -mdl: the model you want to load and use for generating txt. Defaults to "model_278".


    E.g., 
        python src/generate_text.py -t "this news article is horrible but" -n 20

@MinaAlmasi
'''

# to import model 
from tensorflow import keras

# system tools 
from pathlib import Path
import argparse

# custom module
from language_mdl.model_fns import generate_text
from language_mdl.tokenizer_saving import load_tokenizer

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-t", "--seed_text", help = "input text you want to generate from", type = str, default = "this news article is great but")
    parser.add_argument("-n", "--next_words", help = "n words you want to generate after seed_text", type = int, default=5)
    parser.add_argument("-mdl","--model_name", help="model you want to use for generating text e.g., 'model_278'", type=str, default = "model_278")

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    # load args 
    args = input_parse()

    # define modelfolder path
    path = Path(__file__) 
    modelpath = path.parents[1] / "models" / args.model_name

    # load model 
    model = keras.models.load_model(modelpath / f"{args.model_name}.h5")

    # get max_sequence_length 
    max_sequence_len = int(args.model_name.split("_")[1]) # split model name e.g., model_278 [model, 278] and choose 2nd element

    # load tokenizer 
    tokenizer = load_tokenizer(modelpath / f"tokenizer_{max_sequence_len}.pickle") # max sequence length in name

    # generate text
    text = generate_text(args.seed_text, args.next_words, model, tokenizer, max_sequence_len)

    # print generated text ! 
    print(text)

if __name__ == "__main__":
    main()
