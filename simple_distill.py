import os
from datetime import datetime
import argparse
import csv
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchtext import data
from torchtext.vocab import pretrained_aliases, Vocab

from tqdm import tqdm, trange

from transformers import (AdamW, WarmupLinearSchedule, BertConfig, BertForSequenceClassification, BertTokenizer)

from tensorboardX import SummaryWriter

import spacy

spacy_en = spacy.load("en")

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

class Trainer():
    def __init__(self, model, train_dataset, fasttext_pad_token, bert_pad_token, device,
        teacher=None, teacher_alpha=0.0, teacher_loss = "mse",
        val_dataset=None, val_interval=1,
        gradient_accumulation_steps = 1, max_grad_norm=1.0,
        warmup_steps=0, batch_size=50, lr=5e-5, weight_decay=0.0):
        # storing
        self.model = model
        self.train_dataset = train_dataset
        self.fasttext_pad_token = fasttext_pad_token
        self.bert_pad_token = bert_pad_token
        self.device = device
        self.teacher = teacher
        self.teacher_alpha = teacher_alpha
        self.teacher_loss = teacher_loss
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        # initialization
        self.student_loss_f = nn.CrossEntropyLoss(reduction="sum")
        if self.teacher_loss == "mse":
            self.teacher_loss_f = nn.MSELoss()
        elif self.teacher_loss == "kl_div":
            self.teacher_loss_f = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train_dataloader = data.Iterator(self.train_dataset, self.batch_size, train=True, device=self.device)
        if self.val_dataset is not None:
            self.val_dataloader = data.Iterator(self.val_dataset, self.batch_size, train=False, sort_key=lambda x: len(x.fasttext), device=self.device)
        else:
            self.val_dataloader = None
        self.tb_loss = 0
        self.tb_s_loss = 0
        self.tb_t_loss = 0
    def train_step(self, batch, max_steps):
        fasttext_tokens, bert_tokens, labels, length, attention_mask = batch
        self.model.train()
        s_logits = self.model(fasttext_tokens, length)
        s_loss = self.student_loss_f(s_logits, labels) / labels.size(0) # like batchmean
        loss = s_loss
        if self.teacher is not None and self.teacher_alpha > 0.0:
            with torch.no_grad():
                self.teacher.eval()
                t_logits = self.teacher(bert_tokens, attention_mask=attention_mask)[0]
            if self.teacher_loss == "mse":
                t_loss = self.teacher_loss_f(s_logits, t_logits)
            elif self.teacher_loss == "kl_div":
                t_loss = self.teacher_loss_f(
                    F.log_softmax(s_logits / self.temperature, dim=-1),
                    F.softmax(t_logits / self.temperature, dim=-1)
                ) / self.temperature**2
            loss = (1.0 - self.teacher_alpha) * s_loss + self.teacher_alpha * t_loss
        self.tb_loss += loss.item()
        self.tb_s_loss += s_loss.item()
        self.tb_t_loss += t_loss.item() if (self.teacher is not None and self.teacher_alpha > 0.0) else 0.0
        loss.backward()
        if (self.training_step + 1) % self.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], self.global_step)
            self.tb_writer.add_scalar("loss", self.tb_loss / self.gradient_accumulation_steps, self.global_step)
            self.tb_writer.add_scalar("student_loss", self.tb_s_loss / self.gradient_accumulation_steps, self.global_step)
            self.tb_writer.add_scalar("teacher_loss", self.tb_t_loss / self.gradient_accumulation_steps, self.global_step)
            self.tb_loss = 0
            self.tb_s_loss = 0
            self.tb_t_loss = 0
            self.global_step += 1
        if self.val_dataset and (self.global_step + 1) % self.val_interval == 0:
            results = self.evaluate()
            print(results)
            for k, v in results.items():
                self.tb_writer.add_scalar("val_" + k, v, self.global_step)
        self.training_step += 1
    def train(self, epochs=1, max_steps=-1):
        self.global_step = 0
        self.training_step = 0
        total_steps = epochs * len(self.train_dataset) // self.batch_size
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/100, max_lr=self.lr,
            step_size_up=max(1, self.warmup_steps), step_size_down=(total_steps - self.warmup_steps),
            cycle_momentum=False)
        self.tb_writer = SummaryWriter()
        training_it = trange(epochs, desc="Training")
        for epoch in training_it:
            data_it = iter(self.train_dataloader)
            data_it = tqdm(data_it, desc="Epoch %d" % epoch, total=len(self.train_dataset) // self.batch_size)
            for batch in data_it:
                batch = self.process_batch(batch)
                self.train_step(batch, max_steps)
            if max_steps > 0 and self.global_step >= max_steps:
                training_it.close()
                break
    def evaluate(self, eval_teacher=False):
        if eval_teacher:
            self.teacher.eval()
        else:
            self.model.eval()
        val_loss = val_accuracy = 0.0
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        data_it = iter(self.val_dataloader)
        data_it = tqdm(data_it, desc="Evaluation", total=len(self.val_dataset) // self.batch_size)
        for batch in data_it:
            fasttext_tokens, bert_tokens, labels, length, attention_mask = self.process_batch(batch)
            with torch.no_grad():
                if eval_teacher:
                    output = self.teacher(bert_tokens, attention_mask=attention_mask)[0]
                else:
                    output = self.model(fasttext_tokens, length)
                loss = loss_func(output, labels)
                val_loss += loss.item()
                val_accuracy += (output.argmax(dim=-1) == labels).sum().item()
        data_it.close()
        val_loss /= len(self.val_dataset)
        val_accuracy /= len(self.val_dataset)
        return {
            "loss": val_loss,
            "perplexity": np.exp(val_loss),
            "accuracy": val_accuracy
        }
    def process_batch(self, batch):
        fasttext_tokens, bert_tokens, labels = batch.fasttext, batch.bert, batch.label
        length = torch.empty(fasttext_tokens.size(1), dtype=torch.int64).to(labels.device)
        for idx in range(fasttext_tokens.size(1)):
            rg = torch.arange(fasttext_tokens.size(0), device=self.device)
            mask = (fasttext_tokens[:, idx] != self.fasttext_pad_token).type(torch.int64)
            length[idx] = (rg * mask).argmax() + 1
        attention_mask = torch.empty(bert_tokens.size()).to(labels.device)
        for idx in range(bert_tokens.size(0)):
            attention_mask[idx] = (bert_tokens[idx] != self.bert_pad_token).type(attention_mask.type())
        return fasttext_tokens, bert_tokens, labels, length, attention_mask

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

class BertVocab:
    UNK = '<unk>'
    def __init__(self, stoi):
        self.stoi = {}
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
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
        bert_model = BertForSequenceClassification.from_pretrained("./bert_tuned_weights").to(device)
    else:
        bert_model = None
    bert_tokenizer = BertTokenizer.from_pretrained("./bert_tuned_weights", do_lower_case=True)
    train_dataset, valid_dataset, vocab_data = load_data(args.data_dir, bert_tokenizer)

    fasttext_vocab = vocab_data["fasttext_vocab"]
    student_model = BiLSTMClassifier(2, len(fasttext_vocab.itos), fasttext_vocab.vectors.shape[-1],
        lstm_hidden_size=400, classif_hidden_size=800, dropout_rate=0.15, use_multichannel_embedding=True).to(device)
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