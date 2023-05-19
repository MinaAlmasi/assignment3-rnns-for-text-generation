# Language modelling and text generation using RNNs

This repository forms assignment 3 in the subject Language Analytics, Cultural Data Science, F2023. The assignment description can be found [here](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi/blob/main/assignment-desc.md). 

The code is written by **Mina Almasi** (202005465) although some code is adapted from code developed in class (see also [Code Reuse](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi#code-reuse)).

The repository utilizes the comments from the [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) for training a LSTM (RNN) to do text generation. This process involves preprocessing the dataset, training and saving a LSTM, and finally generating text. 

The dataset contains just above 2 million comments, but a model was trained using **only 1000 comments** due to computational limitations (see [Results](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi/tree/main#results) for model card and example of text generation). **It should be noted that the repository is designed to be able to run on the *entire* dataset**

## Reproducibility
To reproduce the model training with a 1000 comments and/or generate text, follow the instructions in the [Pipeline](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi#pipeline) section. This section also contains information on how to train a model with all 2M comments.

**NB! Be aware that training the model is computationally heavy. Cloud computing (e.g., UCloud) with high amounts of ram (or a good GPU) is encouraged.** 

## Project Structure
The repository is structured as such:
```
├── README.md
├── assignment-desc.md
├── data                             <---   place data here ! 
│   └── README.md
├── models                           <---   new models saved here 
│   └── model_278
│       ├── card_model_278.txt       <---   model summary
│       ├── loss_model_278.png       <---   plot of model curve
│       ├── model_278.h5             <---   model (with max sequence length of 278)
│       └── tokenizer_278.pickle     <---   tokenizer
├── requirements.txt                 
├── setup.sh                         <---   creates virtual env, install necessary reqs (from requirements.txt)
├── src
│   ├── data_processing              
│   │   ├── load_data.py             <---   functions to load all data (w. multiprocessing)
│   │   └── process_data.py          <---   functions to clean, tokenize, create & pad input sequences
│   ├── language_mdl
│   │   ├── model_fns.py             <---   functions to create model, generate text, save model card and loss curve
│   │   └── tokenizer_saving.py      <---   functions to save and load tokenizer
│   ├── generate_text.py             <---   script to generate text
│   └── train_model.py               <---   script to train model
├── train.sh                         <---   RUN to train model with subset of 1000 comments for 50 epochs
└── utils
    └── custom_logging.py            <---   custom logger to display user msg
```

## Pipeline 
This pipeline was built on Ubuntu ([UCloud](https://cloud.sdu.dk/)). 

### Setup
First, please download the [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) and place all files in the ```data``` folder. Then, run the ```setup.sh``` in the terminal:
```
bash setup.sh
```
Running ```setup.sh``` will create a virtual environment (```env```) and install the neccesary packages within it. 

### Training the LSTM 
To train the model, you can run the ```train.sh``` script: 
```
bash train.sh
```
**NB! This will train a model on only the first 1000 comments for 50 epocs with an embedding layer of size 10 and a hidden layer of size 30.**

### Custom training of the LSTM
If you want to run the data on a larger subset or different model setup, you can run the script ```train_model.py``` with additional arguments: 

| Arg          | Description                         | Default       |
| :---         |:---                                 |:---           |
| ```-n```     | size of data subset                 | None <br /> (i.e., all 2M comments) |
| ```-el```    | size of embedding layer in LSTM     | 10            |
| ```-hl```    | size of hidden layer in LSTM        | 30            |
| ```-e```     | number of epochs for model training | 50            |

For instance:
```
python src/train_model.py -n 10000 -el 100 -hl 60 -e 100
```
In the case above, the model will train on a subset of 10,000 comments for 100 epochs with an embedding layer of size 100 and a hidden layer of size 60. 

**Note that you need to activate the virtual environment ```env``` prior to running the script:**
```
# activate env
source ./env/bin/activate

# run script 
python src/train_model.py  -n 10000 -el 100 -hl 60 -e 100

# deactivate env
deactivate
```

### Text generation
You can generate text by running the python script ```generate_text.py``` with the virtual ```env``` activated (by running ```source ./env/bin/activate```, see also [Custom training](https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi/tree/main#custom-training-of-the-lstm)).
```
python src/generate_text.py
```
By default, this will generate a ```5-word``` continuation to the sentence ```"this news article is great but"```

You can specify also your own text ```-t``` and decide how many words to generate with the ```-n``` parameter:
```
python src/generate_text.py -t "the election was great" -n 20
```

## Results
### Model Card
The resulting model card from the model that is run with ```bash train.sh```:
```
Model with a max sequence length of 278, trained on 1000 rows of data 
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 277, 10)           98650     
                                                                 
 lstm (LSTM)                 (None, 30)                4920      
                                                                 
 dropout (Dropout)           (None, 30)                0         
                                                                 
 dense (Dense)               (None, 9865)              305815    
                                                                 
=================================================================
Total params: 409,385
Trainable params: 409,385
Non-trainable params: 0
_________________________________________________________________

```
### Loss Curve
<p align="left">
  <img width=60% height=60% src="https://github.com/AU-CDS/assignment-3---rnns-for-text-generation-MinaAlmasi/blob/main/models/model_278/loss_model_278.png">
</p>

The loss curve shows a loss that is decreasing but not one that has hit a plateau. This suggests that the model should have trained for more than ```50``` epochs. 

### Example Generation
When tasked with producing a 5-word continuation of the sentence ```"this news article is great but"```, the model outputs: 
```
this news article is great but the gop is a security
``` 
This is somewhat intelligible, but the model breaks down when it has to produce 10 additional words (```15-word continuation```):
```
this news article is great but the gop is a security and the gop is a security and the gop is
```

### Remarks on the performance
Along with the loss curve, the repetitive text generation suggests that the model training was not entirely succesful. 

Model performance may be improved when running on the entire dataset. Possible improvements may also come from adding more hidden layers, a larger embedding layer (or [pre-trained GloVe embeddings](https://nlp.stanford.edu/projects/glove/)) along with more epochs. However, all these options require greater computational power than what was available at the time of making this repository. 

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk


### Code Reuse 
Some code was either developed in a previous assignment in the courses *language analytics* or *visual analytics* or taken from class notebooks and adapted. Whenever this is the case, it will be stated in the particular scripts as a ```#comment``` or in the script docstring. 
