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

from trainer import BertTrainer
from utils import set_seed, load_data, BertVocab

def save_bert(model, tokenizer, config, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    bert_config.save_pretrained(output_dir)
    bert_model.save_pretrained(output_dir)
    bert_tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where to save the model.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "warmup", "cyclic"],
        help="Schedule to use for the learning rate. Choices are: constant, linear warmup & decay, cyclic.")
    parser.add_argument("--warmup_steps", type=int, default=0,
        help="Warmup steps for the 'warmup' learning rate schedule. Ignored otherwise.")
    parser.add_argument("--epochs_per_cycle", type=int, default=1,
        help="Epochs per cycle for the 'cyclic' learning rate schedule. Ignored otherwise.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=-1)
    parser.add_argument("--cache_dir", type=str, help="Custom cache for transformer models.")
    args = parser.parse_args()

    if args.lr_schedule == "constant":
        args.lr_schedule = None
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    bert_config = BertConfig.from_pretrained("bert-large-uncased", cache_dir=args.cache_dir)
    bert_model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=bert_config, cache_dir=args.cache_dir).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, cache_dir=args.cache_dir)
    train_dataset, valid_dataset, _ = load_data(args.data_dir, bert_tokenizer.tokenize,
        vocab=BertVocab(bert_tokenizer.vocab), batch_first=True)
    
    trainer = BertTrainer(bert_model, device,
        loss="cross_entropy",
        train_dataset=train_dataset,
        val_dataset=valid_dataset, val_interval=250,
        checkpt_callback=lambda m, step: save_bert(m, bert_tokenizer, bert_config, os.path.join(args.output_dir, "checkpt_%d" % step)),
        checkpt_interval=args.checkpoint_interval,
        batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr)
    if args.do_train:
        trainer.train(args.epochs, schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps, epochs_per_cycle=args.epochs_per_cycle)

    print("Evaluating model:")
    print(trainer.evaluate())

    save_bert(bert_model, bert_tokenizer, bert_config, args.output_dir)
