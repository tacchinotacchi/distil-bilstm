import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)

from trainer import LSTMTrainer
from utils import set_seed, load_data, spacy_tokenizer

class MultiChannelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, filters_size=64, filters=[2, 4, 6], dropout_rate=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.filters_size = filters_size
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.conv1 = nn.ModuleList([
            nn.Conv1d(self.embed_size, filters_size, kernel_size=f, padding=f//2)
            for f in filters
        ])
        self.act = nn.Sequential(
            nn.ReLU(inplace=True),
            #nn.Dropout(p=dropout_rate)
        )
    def init_embedding(self, weight):
        self.embedding.weight = nn.Parameter(weight.to(self.embedding.weight.device))
    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.embedding(x).transpose(1, 2)
        channels = []
        for c in self.conv1:
            channels.append(c(x))
        x = F.relu(torch.cat(channels, 1))
        x = x.transpose(1, 2).transpose(0, 1)
        return x   

class BiLSTMClassifier(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_size, lstm_hidden_size, classif_hidden_size,
        lstm_layers=1, dropout_rate=0.0, use_multichannel_embedding=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_hidden_size = lstm_hidden_size
        self.use_multichannel_embedding = use_multichannel_embedding
        if self.use_multichannel_embedding:
            self.embedding = MultiChannelEmbedding(self.vocab_size, embed_size, dropout_rate=dropout_rate)
            self.embed_size = len(self.embedding.filters) * self.embedding.filters_size
        else:
            self.embedding = nn.Embedding(self.vocab_size, embed_size)
            self.embed_size = embed_size
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_size, lstm_layers, bidirectional=True, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, classif_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(classif_hidden_size, num_classes)
        )
    def init_embedding(self, weight):
        if self.use_multichannel_embedding:
            self.embedding.init_embedding(weight)
        else:
            self.embedding.weight = nn.Parameter(weight.to(self.embedding.weight.device))
    def forward(self, seq, length):
        # Sort batch
        seq_size, batch_size = seq.size(0), seq.size(1)
        length_perm = (-length).argsort()
        length_perm_inv = length_perm.argsort()
        seq = torch.gather(seq, 1, length_perm[None, :].expand(seq_size, batch_size))
        length = torch.gather(length, 0, length_perm)
        # Pack sequence
        seq = self.embedding(seq)
        seq = pack_padded_sequence(seq, length)
        # Send through LSTM
        features, hidden_states = self.lstm(seq)
        # Unpack sequence
        features = pad_packed_sequence(features)[0]
        # Separate last dimension into forward/backward features
        features = features.view(seq_size, batch_size, 2, -1)
        # Index to get forward and backward features and concatenate
        # Gather last word for each sequence
        last_indexes = (length - 1)[None, :, None, None].expand((1, batch_size, 2, features.size(-1)))
        forward_features = torch.gather(features, 0, last_indexes)
        # Squeeze seq dimension, take forward features
        forward_features = forward_features[0, :, 0]
        # Take first word, backward features
        backward_features = features[0, :, 1]
        features = torch.cat((forward_features, backward_features), -1)
        # Send through classifier
        logits = self.classifier(features)
        # Invert batch permutation
        logits = torch.gather(logits, 0, length_perm_inv[:, None].expand((batch_size, logits.size(-1))))
        return logits, hidden_states

def save_bilstm(model, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", type=str)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--epochs_per_cycle", type=int, default=1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_dataset, valid_dataset, vocab = load_data(args.data_dir, spacy_tokenizer, augmented=args.augmented)

    model = BiLSTMClassifier(2, len(vocab.itos), vocab.vectors.shape[-1],
        lstm_hidden_size=300, classif_hidden_size=400, dropout_rate=0.0).to(device)
    model.init_embedding(vocab.vectors.to(device))
    
    trainer = LSTMTrainer(model, "cross_entropy", vocab.stoi["<pad>"], device,
        train_dataset=train_dataset, val_dataset=valid_dataset, val_interval=250,
        checkpt_callback=lambda m, step: save_bilstm(m, os.path.join(args.output_dir, "checkpt_%d" % step)), checkpt_interval=250,
        batch_size=args.batch_size, lr=args.lr)
    
    if args.do_train:
        trainer.train(args.epochs, schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps, epochs_per_cycle=args.epochs_per_cycle)
    
    print("Evaluating model:")
    print(trainer.evaluate())

    save_bilstm(model, args.output_dir)

