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

Read the [Structure](#structure) section below. It specifies what each folder and file does in this project.  
If there is any doubt, look into the code as well.

#### 3. Finish the tutorial

The only two files you need to modify are `src/train/model.py` and `src/train/train.py`.

1. `src/train/model.py`
    1. This file defines the LSTM model that takes sequences as input and predicts the next tokens.
    2. Locate `STEP XXX` in the file. Fill in the missing codes by reading the descriptions.
    3. Test it by running `python3 -m src.train.train corpus/corpus_min.txt`. If everything is correct, no errors will be reported. However, the loss will not be reducing.
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
- Train on the original large corpus `corpus/corpus.txt`
- Increase embedding dimension
- Increase hidden dimension
- Increase the number of epochs

## Example Output

The following example outputs were trained on the large corpus with the following hyperparameters:
- sequence length = 50
- batch size = 32
- embedding dimension = 256
- hidden dimension = 256
- learning rate = 0.0001
- dropout = 0.2
- epoch = 30

The training time was 5~6 hours on 2 GHz Intel Core i5 CPU.

```
在战争中否坚决地实行和全体生活的形势是错误的，那一世界有特点的人们想不另一时，
愿意的指导、非英时、半成大党的估计。按照这个人，若不重视显极，而是使他们取得
胜利。我也一定要君子使国家吗？可以白有的大子加制，使使他服长。难道这怎么能短
良呢？则我早上怎这么样？说得到太的觉多了没才会名季厌。恶—那样里来不会自六方）
，不奏怨恨，我敢不敬，言语他的推子。君子恭敬仁义。帝国主义已经：说多了，不可
见变愿意十五，即后，一战不能打干，就自以只是在那类里的自线之养。

智的诸侯了，天下的交朝吗？有礼有制度，然后却是殷容易听了。马克思列宁主义者，
要学习端木不是教别的方法，不可以说出发死的。了符合的主义形规定的就是无不理的，
还有一问当，最简如见同同志同民族领导、干部、活化，相间工互来关头党的整风。但
是还每一个整一个新干部，要看教育内部同统权力更等、干部其、活庭和巩固群众。这
不是尊重那个问题，就要说弄个根据，不作任何由统一的斗争。但是在我国现在资本主
义的剥削和共产党，不是很好地接受社会主义的工作方向。每一个基本作用外，结果有
一个专于战役它。党的政治任务对于行动；（要有几先之礼，几然年，不允动地也会必
要要使拒主观愿意。十岁伟大还有中国历史这种具体这要的。

争，我们会主义和平国，而没有把国的人和厌恶。这样的人不宽易。君子，却没有正确
的思德，就是学会了一次来中这样的吗？一个有人一天要看到，一个同志也不可以认真，
如果不任用所保留的人，去用仁德守住，就是孝从鬼神神，到这样微了。在练兵，都思
考情齐智百了，就不说，该容易作等；学斗争取青，不给以那就不去，全年的才能。以
发挥为杀王的不能不会得彻底，他来不可以做；不而不上与，拿着我们，也将他打民主
观。一定要说是不安的；言论周别的人对亲下，不吃肉，不吃。当然而推举，拿受蒋介
石的圣人。宰查回果是什么事了，可以让他自己高兴的事，小人批评。不说话，这是反
而不是孝悌。
```

## Structure

```
├── corpus                                          # Raw & parsed corpus
│   ├── corpus.txt                                      # Original mixed corpus file
│   └── corpus_min.txt                                    # Minimized corpus for fast training
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
