import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Define model layers
        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, n_vocab)

    def forward(self, seq_in):
        # seq_in: batch_size x seq_length

        # Map each input character to its embedding
        embeddings = self.embeddings(seq_in)     # batch_size x seq_length x embedding_dim

        # Feed the embeddings through the LSTM layer
        # Retrieve the last hidden state as the encoded representation of the sequence
        _, (ht, _) = self.lstm(embeddings)       # Each timestep outputs 1 hidden_state
                                                 # ht = hidden state for the last timestep
                                                 #    = (num_layers * num_directions) x batch_size x hidden_dim
                                                 #    = 1 x batch_size x hidden_dim
        ht = ht[0]                               # batch_size x hidden_dim (we don't need the first dimension)

        # Feed the last hidden state through the fully-connected layer
        # Retrieve the probability distribution of each character being the next token
        out = self.hidden2out(ht)                # batch_size x n_vocab

        return out
