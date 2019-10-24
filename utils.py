import os
import re
import csv
from collections import OrderedDict

import numpy as np
import torch
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab

import spacy
spacy_en = spacy.load("en")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class BertVocab:
    UNK = '<unk>'
    def __init__(self, stoi):
        self.stoi = OrderedDict()
        pattern = re.compile(r"\[(.*)\]")
        for s, idx in stoi.items():
            s = s.lower()
            m = pattern.match(s)
            if m:
                content = m.group(1)
                s = "<%s>" % content
            self.stoi[s] = idx
        self.unk_index = self.stoi[BertVocab.UNK]
        self.itos = [(s, idx) for s, idx in self.stoi.items()]
        self.itos.sort(key=lambda x: x[1])
        self.itos = [s for (s, idx) in self.itos]
    def _default_unk_index(self):
        return self.unk_index
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(BertVocab.UNK))
    def __len__(self):
        return len(self.itos)

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_tsv(path, skip_header=True):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader)
        data = [row for row in reader]
    return data

def load_data(data_dir, tokenizer, bert_vocab=None, batch_first=False, augmented=False):
    text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=batch_first)
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float32 if augmented else torch.long)
    train_dataset, valid_dataset = data.TabularDataset.splits(
        path=data_dir, train="augmented.tsv" if augmented else "train.tsv", validation="dev.tsv",
        format="tsv", skip_header=True,
        fields=[("text", text_field), ("label", label_field)])
    if bert_vocab is None:
        vectors = pretrained_aliases["fasttext.en.300d"](cache=".cache/")
        text_field.build_vocab(train_dataset, vectors=vectors)
    else:
        # use bert tokenizer's vocab if supplied
        text_field.vocab = BertVocab(bert_vocab)
    return train_dataset, valid_dataset, text_field.vocab