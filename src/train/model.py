import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        '''
            STEP 1.1: Define model layers
                Define three layers in our neural network model:
                    1. Embedding layer, which maps `n_vocab` characters to its corresponding embedding vector.
                       Look up official doc for `torch.nn.Embedding`.
                    2. LSTM layer, which takes in input with shape `batch x seq_len x embedding_dim`
                       and produces a hidden vector of size `hidden_dim` for each timestep.
                       Look up official doc for `torch.nn.LSTM`.
                    3. Fully-connected layer (linear layer), which takes a hidden vector
                       and outputs a score for each character in the vocabulary.
                       Look up official doc for `torch.nn.Linear`.
                Replace `None` within CODE STARTS and CODE ENDS with your answers.
        '''
        # CODE STARTS
        self.embeddings = None
        self.lstm = None
        self.hidden2out = None
        # CODE ENDS

    def forward(self, seq_in):
        # seq_in: batch_size x seq_length

        '''
            STEP 1.2: Map each input character to its embedding
                The original input `seq_in` contains sequences of integers that represent the id of each word in the vocabulary.
                Map them to their corresponding embedding vector.
                Look up official doc for `torch.nn.Embedding`.

                Replace `None` within CODE STARTS and CODE ENDS with your answers.
        '''
        # CODE STARTS
        embeddings = None
        # CODE ENDS

        # Feed the embeddings through the LSTM layer
        # Retrieve the last hidden state as the encoded representation of the sequence
        _, (ht, _) = self.lstm(embeddings)       # Each timestep outputs 1 hidden_state
                                                 # ht = hidden state for the last timestep
                                                 #    = (num_layers * num_directions) x batch_size x hidden_dim
                                                 #    = 1 x batch_size x hidden_dim
        ht = ht[0]                               # batch_size x hidden_dim (we don't need the first dimension)

        '''
            STEP 1.3: Feed the last hidden state through the fully-connected layer.
                      Retrieve the probability distribution of each character being the next token.
                Now we have the hidden vectors of the batch `ht`.
                Feed it through our fully-connected layer to retrieve a scalar score for each hidden vector.
                Look up official doc for `torch.nn.Linear`.

                Replace `None` within CODE STARTS and CODE ENDS with your answers.
        '''
        # CODE STARTS
        out = None
        # CODE ENDS

        return out
