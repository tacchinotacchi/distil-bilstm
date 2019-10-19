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

from trainer import Trainer
from utils import set_seed, load_data, BertVocab

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
        # sort batch
        seq_size, batch_size = seq.size(0), seq.size(1)
        length_perm = (-length).argsort()
        length_perm_inv = length_perm.argsort()
        seq = torch.gather(seq, 1, length_perm[None, :].expand(seq_size, batch_size))
        length = torch.gather(length, 0, length_perm)
        # compute
        seq = self.embedding(seq)
        seq = pack_padded_sequence(seq, length)
        features = self.lstm(seq)[0]
        features = pad_packed_sequence(features)[0]
        features = features.view(seq_size, batch_size, 2, -1)
        # index to get forward and backward features
        last_indexes = (length - 1)[None, :, None, None].expand((1, batch_size, 2, features.size(-1)))
        forward_features = torch.gather(features, 0, last_indexes)
        forward_features = forward_features[0, :, 0]
        backward_features = features[0, :, 1]
        features = torch.cat((forward_features, backward_features), -1)
        # send through classifier
        logits = self.classifier(features)
        # invert batch order
        logits = torch.gather(logits, 0, length_perm_inv[:, None].expand((batch_size, logits.size(-1))))
        return logits

def save_bilstm(model, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--teacher_alpha", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.teacher_alpha > 0.0:
        bert_model = BertForSequenceClassification.from_pretrained("./bert_large_tuned").to(device)
    else:
        bert_model = None
    bert_tokenizer = BertTokenizer.from_pretrained("./bert_large_tuned", do_lower_case=True)
    train_dataset, valid_dataset, vocab_data = load_data(args.data_dir, bert_tokenizer)

    fasttext_vocab = vocab_data["fasttext_vocab"]
    student_model = BiLSTMClassifier(2, len(fasttext_vocab.itos), fasttext_vocab.vectors.shape[-1],
        lstm_hidden_size=400, classif_hidden_size=800, dropout_rate=0.15).to(device)
    fasttext_emb = fasttext_vocab.vectors.to(device)
    student_model.init_embedding(fasttext_emb)
    
    trainer = Trainer(student_model, train_dataset,
        vocab_data["fasttext_vocab"].stoi["<pad>"], vocab_data["bert_vocab"].stoi["<pad>"], device,
        val_dataset=valid_dataset, val_interval=100,
        batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup_steps,
        teacher=bert_model, teacher_alpha=args.teacher_alpha)
    if bert_model is not None:
        print("Evaluating teacher:")
        print(trainer.evaluate(eval_teacher=True))
    if args.do_train:
        trainer.train(args.epochs)
    print("Evaluating model:")
    print(trainer.evaluate())

    save_bilstm(student_model, args.output_dir)

