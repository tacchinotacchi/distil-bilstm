import os
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
        for s, idx in stoi.items():
            if s == "[UNK]":
                s = "<unk>"
            elif s == "[PAD]":
                s = "<pad>"
            self.stoi[s] = idx
        self.unk_index = self.stoi[BertVocab.UNK]
        self.itos = [(s, idx) for s, idx in self.stoi.items()]
        self.itos.sort(key=lambda x: x[1])
        self.itos = [s for (s, idx) in self.itos]
    def _default_unk_index(self):
        return self.unk_index
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(BertVocab.UNK))
    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs
    def __setstate__(self, state):
        if state.get("unk_index", None) is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)
    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True
    def __len__(self):
        return len(self.itos)

def spacy_tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_tsv(path, row_permutation=None, conversions=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        data = []
        for row in reader:
            if row_permutation is not None:
                row = [row[idx] for idx in row_permutation]
            if conversions is not None:
                row = [
                    c(row[idx]) if c is not None else row[idx]
                    for idx, c in enumerate(conversions)
                ]
            data.append(row)
    return data

def load_data(data_dir, bert_tokenizer):
    splits = [
        load_tsv(os.path.join(data_dir, split_file), row_permutation=(0, 0, 1), conversions=[None, None, int])
        for split_file in ("train.tsv", "dev.tsv")
    ]
    fasttext_field = data.Field(sequential=True, tokenize=spacy_tokenizer, lower=True)
    bert_field = data.Field(sequential=True, tokenize=bert_tokenizer.tokenize, lower=True, batch_first=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    fields = [("fasttext", fasttext_field), ("bert", bert_field), ("label", label_field)]
    examples = [
        [data.Example.fromCSV(item, fields) for item in split]
        for split in splits
    ]
    train_dataset, valid_dataset = [
        data.Dataset(split, fields)
        for split in examples
    ]
    # set up fasttext field
    fasttext_vectors = pretrained_aliases["fasttext.en.300d"](cache=".cache/")
    fasttext_field.build_vocab(train_dataset, vectors=fasttext_vectors)
    # set up bert field
    bert_field.vocab = BertVocab(bert_tokenizer.vocab)
    #fasttext_pad_token = fasttext_field.stoi["<pad>"]
    #bert_pad_token = bert_field.vocab.stoi["<pad>"]
    return train_dataset, valid_dataset, {
        "fasttext_vocab": fasttext_field.vocab,
        "bert_vocab": bert_field.vocab
    }