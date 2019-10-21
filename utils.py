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
                print(content)
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
    def __getstate__(self):
        # TODO what does this do?
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

def infer(model, vocab, sentence):
    seq = spacy_tokenizer(sentence)
    with torch.no_grad():
        seq = torch.tensor([[vocab.stoi[t.lower()]] for t in seq], dtype=torch.int64, device=next(model.parameters()).device)
        length = torch.tensor([seq.size(0)], dtype=torch.int64, device=seq.device)
        output = model(seq, length)
    return output

def load_data(data_dir, bert_tokenizer=None, augmented=False):
    fasttext_field = data.Field(sequential=True, tokenize=spacy_tokenizer, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    if bert_tokenizer is not None:
        bert_field = data.Field(sequential=True, tokenize=bert_tokenizer.tokenize, lower=True, batch_first=True)
        perm, convs = (0, 0, 1), (None, None, int)
        fields = [("fasttext", fasttext_field), ("bert", bert_field), ("label", label_field)]
    else:
        perm, convs = (0, 1), (None, int)
        fields = [("fasttext", fasttext_field), ("label", label_field)]
    splits = [
        load_tsv(os.path.join(data_dir, split_file), row_permutation=perm, conversions=convs)
        for split_file in (
            ("augmented.tsv" if augmented else "train.tsv"),
            "dev.tsv"
        )
    ]
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
    if bert_tokenizer is not None:
        bert_field.vocab = BertVocab(bert_tokenizer.vocab)
    return train_dataset, valid_dataset, {
        "fasttext_vocab": fasttext_field.vocab,
        "bert_vocab": bert_field.vocab if bert_tokenizer is not None else None
    }