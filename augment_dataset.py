import os
import argparse

import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torchtext import data
from transformers import BertForSequenceClassification, BertTokenizer
from trainer import BertTrainer
from utils import spacy_en, load_tsv, set_seed, BertVocab

def build_pos_dict(sentences):
    pos_dict = {}
    for sentence in sentences:
        for word in sentence:   
            pos_tag = word.pos
            if pos_tag not in pos_dict:
                pos_dict[pos_tag] = []
            if word.text.lower() not in pos_dict[pos_tag]:
                pos_dict[pos_tag].append(word.text.lower())
    return pos_dict

mask_token = "<mask>"

def make_sample(input_sentence, pos_dict, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    sentence = []
    for word in input_sentence:
        # Apply single token masking or POS-guided replacement
        u = np.random.uniform()
        if u < p_mask:
            sentence.append(mask_token)
        elif u < (p_mask + p_pos):
            same_pos = pos_dict[word.pos]
            # Pick from list of words with same POS tag
            sentence.append(np.random.choice(same_pos))
        else:
            sentence.append(word.text.lower())
    # Apply n-gram sampling
    if len(sentence) > 2 and np.random.uniform() < p_ng:
        n = min(np.random.choice(range(1, 5+1)), len(sentence) - 1)
        start = np.random.choice(len(sentence) - n)
        for idx in range(start, start + n):
            sentence[idx] = mask_token
    return sentence

    
def augmentation(sentences, pos_dict, n_iter=20, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    augmented = []
    for sentence in tqdm(sentences, "Generation"):
        samples = [[word.text.lower() for word in sentence]]
        for _ in range(n_iter):
            new_sample = make_sample(sentence, pos_dict, p_mask, p_pos, p_ng, max_ng)
            if new_sample not in samples:
                samples.append(new_sample)
        augmented.extend(samples)
    return augmented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dataset.")
    parser.add_argument("--output", type=str, required=True, help="Output dataset.")
    parser.add_argument("--model", type=str, required=True, help="Model to use to generate the labels for the augmented dataset.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load original tsv file
    input_tsv = load_tsv(args.input)
    sentences = [spacy_en(text) for text, _ in tqdm(input_tsv, desc="Loading dataset")]

    # build lists of words indexes by POS tab
    pos_dict = build_pos_dict(sentences)

    # Generate augmented samples
    sentences = augmentation(sentences, pos_dict)

    # Load teacher model
    model = BertForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)

    # Assign labels with teacher
    teacher_field = data.Field(sequential=True, tokenize=tokenizer.tokenize, lower=True, include_lengths=True, batch_first=True)
    fields = [("text", teacher_field)]
    examples = [
        data.Example.fromlist([" ".join(words)], fields) for words in sentences
    ]
    augmented_dataset = data.Dataset(examples, fields)
    teacher_field.vocab = BertVocab(tokenizer.vocab)
    new_labels = BertTrainer(model, "cross_entropy", device, batch_size=args.batch_size).infer(augmented_dataset)

    # Write to file
    with open(args.output, "w") as f:
        f.write("sentence\tscores\n")
        for sentence, rating in zip(sentences, new_labels):
            text = " ".join(sentence)
            f.write("%s\t%.6f %.6f\n" % (text, *rating))
        
    
