import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .data import parse_corpus, format_data
from .model import Net


'''Helper methods'''
def load_data(path, seq_length, batch_size):
    dataX, dataY, char_to_int, int_to_char, chars = parse_corpus(path, seq_length=seq_length)
    data = format_data(dataX, dataY, n_classes=len(chars), batch_size=batch_size)

    return data, dataX, dataY, char_to_int, int_to_char, chars

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

'''Main training method'''
def train(model, optimizer, epoch, data):
    model.train() # set to training mode

    # Run over a number of batches
    for batch_i, (seq_in, target) in enumerate(data):
        # Clear gradients stored in parameters
        optimizer.zero_grad()

        # Predict output with model
        output = model(seq_in)

        # Calculate cross entropy loss between output and target
        loss = F.cross_entropy(output, target)

        # Backpropagate loss to each parameter in network
        loss.backward()

        # Update parameters with the optimizer
        optimizer.step()

        # Log training status every 10 batches
        if batch_i % 10 == 0:
            print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.item()))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train seq2seq model')
    parser.add_argument('corpus', type=str, metavar='CORPUS',
                        help='training corpus file')
    parser.add_argument('--seq-length', type=int, default=20, metavar='SL',
                        help='input sequence length (default: 20)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS',
                        help='training batch size (default: 16)')
    parser.add_argument('--embedding-dim', type=int, default=50, metavar='ED',
                        help='embedding dimension for characters in corpus (default: 50)')
    parser.add_argument('--hidden-dim', type=int, default=64, metavar='HD',
                        help='hidden state dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--output', type=str, default='model.trc', metavar='OUTPUT',
                        help='output model file (default: model.trc)')
    parser.add_argument('--output-c', type=str, default='corpus.pkl', metavar='OUTPUT-CORPUS',
                        help='output corpus related file (mappings & vocab) (default: corpus.pkl)')
    args = parser.parse_args()

    # Prepare training data
    train_data, dataX, dataY, char_to_int, int_to_char, chars = load_data(args.corpus, seq_length=args.seq_length, batch_size=args.batch_size)

    # Create model
    model = Net(len(chars), args.embedding_dim, args.hidden_dim)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train for a number of epochs
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, train_data)
        torch.save(model, args.output) # save model

    # Save mappings, vocabs
    save_pickle((dataX, char_to_int, int_to_char, chars), args.output_c)
