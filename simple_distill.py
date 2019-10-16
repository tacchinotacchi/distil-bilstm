import os
from datetime import datetime
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab

from tqdm.autonotebook import tqdm, trange

from transformers import (AdamW, WarmupLinearSchedule, BertConfig, BertForSequenceClassification, BertTokenizer)

from tensorboardX import SummaryWriter

import spacy

spacy_en = spacy.load("en")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    bert_model = BertForSequenceClassification.from_pretrained("./bert_tuned_2").to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("./bert_tuned_2", do_lower_case=True)

    splits = [
        load_tsv(os.path.join(args.data_dir, split_file), row_permutation=(0, 0, 1), conversions=[None, None, int])
        for split_file in ("train.tsv", "test.tsv")
    ]
    fasttext_field = data.Field(sequential=True, tokenize=spacy_tokenizer, lower=True)
    bert_field = data.Field(sequential=True, tokenize=bert_tokenizer.tokenize, lower=True)
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
    fasttext_vectors = pretrained_aliases["fasttext.en.300d"](cache=".cache/")
    fasttext_field.build_vocab(train_dataset, vectors=fasttext_vectors)