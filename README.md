# Resurrecting the Dead - Chinese

Text generation system based on a mixed corpus of 《毛澤東語錄》(Quotations From Chairman Mao Tse-Tung) and《論語》(Confucian Analects).

|Framework|Model|Optimizer|
|:-:|:-:|:-:|
| PyTorch 0.4.1 | RNN (LSTM) | Adam |

[穿越時空的偉人：用PyTorch重現偉人們的神經網絡](https://pyliaorachel.github.io/blog/tech/nlp/2017/12/24/resurrecting-the-dead-chinese.html)

## Steps

#### 1. Download project & create virtual environment

```bash
# Clone project
$ git clone https://github.com/pyliaorachel/resurrecting-the-dead-chinese-tutorial.git
$ cd resurrecting-the-dead-chinese-tutorial

# Create virtual environment, Python3 used
$ virtualenv -p python3 .venv
$ source .venv/bin/activate

# Install dependencies
$ pip3 install -r requirements.txt
```

#### 2. Understand the Project

Read the [Structure](#structure) section below. It specifies what each folder and file do in this project.  
If there is any doubt, look into the code as well.

#### 3. Finish the tutorial

The only two files you need to modify are `src/train/model.py` and `src/train/train.py`.

1. `src/train/model.py`
    1. This file defines the LSTM model that takes sequences as input and predicts the next tokens.
    2. Locate `STEP XXX` in the file. Fill in the missing codes by reading the descriptions.
    3. Test it by running `python3 -m src.train.train corpus/corpus_min.txt`. If everything is correct, no errors will be reported.
2. `src/train/train.py`
    1. After finishing `src/train/model.py`, the model is correctly defined but not actually trained. This file implements backpropagation so that the model parameters can be continuously updated and approach optima.
    2. Locate `STEP XXX` in the file. Fill in the missing codes by reading the descriptions.
    3. Test it by running `python3 -m src.train.train corpus/corpus_min.txt`. If everything is correct, no errors will be reported, and you will see the loss gradually reducing.

When you get frustrated, you can take a look at `src/train/model_gold.py` and `src/train/train_gold.py` for the finished implementations.

#### 4. Train the model

After the codes are finished, you can start training the model and go grab some tea.

```bash
$ python3 -m src.train.train corpus/corpus_min.txt
```

Outputs:

- `model.trc`: torch model
- `corpus.pkl`: parsed corpus, mapping, & vocabulary

The model will finish training after 3~10 mins with the default settings, depending on the power of your machine.

#### 5. Text generation

Locate `model.trc` and `corpus.pkl` files. Use them to generate some text that you can read and evaluate.

```bash
$ python3 -m src.generate_text.gen corpus.pkl model.trc 
```

#### 6. (Optional) Fine-Tuning

Try tuning the hyperparameters or using the large corpus to train for better performance.

Find out what hyperparameters you can set with:

```bash
$ python3 -m src.train.train -h
```

Examples you may try:
- Train the original large corpus `corpus/corpus.txt`
- Increase embedding dimension
- Increase hidden dimension
- Increase the number of epochs

## Structure

```
├── corpus                                          # Raw & parsed corpus
│   ├── corpus.txt                                      # Original mixed corpus file
│   └── corpus_min.txt                                    # Minimized corpus for fast training
├── output                                          # Results
│   ├── log                                             # Log files
│   └── model                                           # Pretrained models
│       └── slxx-bsxx-edxx-hdxx-lrxx-drxx-epxx              # seq_length, batch_size, embedding_dim, hidden_dim, 
│                                                           # learning_rate, dropout, epochs
└── src                                             # Source codes
    ├── generate_text                                   # Text generation
    │   └── gen.py                                          # Text generation
    └── train                                           # Model training
        ├── data.py                                         # Parse data
        ├── model.py                                        # Main LSTM model (to be implemented)
        ├── model_gold.py                                   # Main LSTM model (finished)
        ├── train.py                                        # Training (to be implemented)
        └── train_gold.py                                   # Training (finished)
```
