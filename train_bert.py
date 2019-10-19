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

def save_bert(model, tokenizer, config, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    bert_config.save_pretrained(output_dir)
    bert_model.save_pretrained(output_dir)
    bert_tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    bert_config = BertConfig.from_pretrained("bert-large-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=bert_config).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    train_dataset, valid_dataset, vocab_data = load_data(args.data_dir, bert_tokenizer)
    
    trainer = Trainer(bert_model, train_dataset,
        vocab_data["fasttext_vocab"].stoi["<pad>"], vocab_data["bert_vocab"].stoi["<pad>"], device,
        training_tokens="bert",
        val_dataset=valid_dataset, val_interval=500,
        batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.do_train:
        trainer.train(args.epochs)
    print("Evaluating model:")
    print(trainer.evaluate())

    save_bert(bert_model, bert_tokenizer, bert_config, args.output_dir)