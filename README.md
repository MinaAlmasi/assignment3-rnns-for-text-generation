# Language Modelling and Text Generation using RNNs
Repository link: https://github.com/MinaAlmasi/assignment3-rnns-for-text-generation 

This repository forms *assignment 3* by Mina Almasi (202005465) in the subject Language Analytics, Cultural Data Science, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment3-rnns-for-text-generation/blob/main/assignment-desc.md). 

The repository contains code for training an RNN model to generate text (performing next word predictions). This process involves preprocessing the dataset, training and saving the model, and finally generating text (see [Results](https://github.com/MinaAlmasi/assignment3-rnns-for-text-generation#results) for model card and example of text generation). 

## Dataset 
The repository utilizes the [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments). The dataset contains just above 2 million comments from New York Times articles in the time period Jan-May 2017. 

Despite the size of the dataset, a model was trained using ```only 1000 comments``` due to computational limitations. **Importantly, it should be noted that the code is designed to be able to run on the *entire* dataset**.

## Reproducibility
To reproduce the model training with a 1000 comments and/or generate text, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/assignment3-rnns-for-text-generation/tree/main#pipeline) section. This section also contains information on how to train a model with all 2M comments.

NB! Be aware that training the model is computationally heavy. Cloud computing (e.g., [UCloud](https://cloud.sdu.dk/)) with high amounts of ram (or a good GPU) is encouraged.

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
│   ├── modules
│   │   ├── load_data.py             <---   functions to load all data (w. multiprocessing)
│   │   ├── model_fns.py             <---   functions to create model, generate text, save model
│   │   └── process_data.py          <--- 
│   ├── generate_text.py             <---   run to generate text
│   └── train_model.py               <---   run to train model
├── train.sh                         <---   run to train model w. data subset & generate example text
└── utils
    └── custom_logging.py            <---   custom logger to display user msg
```

## Pipeline 
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Setup
First, please download the [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) and place all files in the ```data``` folder. 

Secondly, create a virtual environment (```env```) and install necessary requirements by running:
```
bash setup.sh
```

### Running the Pipeline (Model Training and Generation)
To train the model and generate example text, type: 
```
bash run.sh
```

To simply run the training pipeline, type: 
```
bash train.sh
```

**NB! Note that both options will train a model on only the first 1000 comments for 50 epocs with an embedding layer of size 10 and a hidden layer of size 30.**

### Custom Training of the Model
If you wish to run the data on a larger subset or different model setup, you can run the script ```train_model.py``` with additional arguments:
```
python src/train_model.py -n {SUBSET_DATA} -el {EMBEDDING_LAYER} -hl {HIDDEN_LAYER} -e {EPOCHS}
```


| Arg          | Description                         | Default       |
| :---         |:---                                 |:---           |
| ```-n```     | size of data subset                 | None <br /> (i.e., all 2M comments) |
| ```-el```    | size of embedding layer in model    | 10            |
| ```-hl```    | size of hidden layer in model       | 30            |
| ```-e```     | number of epochs for model training | 50            |

NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)


### Text Generation
You can generate text by running ```generate_text.py``` with the virtual ```env``` activated:
```
python src/generate_text.py
```
By default, this will generate a ```5-word``` continuation to the sentence ```"this news article is great but"```

You can also run your own generation with the custom arguments:
```
python src/generate_text.py -t {TEXT_TO_CONTINUE} -n {N_WORDS_TO_GENERATE} -mdl {MODEL}
```

| <div style="width:130x">Arg</div>             | <div style="width:150px">Description</div>  |  <div style="width:100px">Default</div>                           |
| :---         |:---                                                                      |:---                              |
| ```-t```     | Text that the model should continue (by generating next word predictions)| "this news article is great but" |
| ```-n```     | Number of words to generate                                              | 5                                |
| ```-mdl```   | the model you want to load and use for generating text                   | "model_278"                      |


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
  <img width=60% height=60% src="https://github.com/MinaAlmasi/assignment3-rnns-for-text-generation/blob/main/models/model_278/loss_model_278.png">
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

### Remarks on the Performance
Along with the loss curve, the repetitive text generation suggests that the model training was not entirely succesful. 

Model performance may be improved when running on the entire dataset. Possible improvements may also come from adding more hidden layers, a larger embedding layer (or [pre-trained GloVe embeddings](https://nlp.stanford.edu/projects/glove/)) along with more epochs. However, all these options require greater computational power than what was available at the time of making this repository. 

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk
