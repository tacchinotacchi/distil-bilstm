import os
import argparse

import numpy as np
from tqdm import tqdm

from utils import spacy_en, load_tsv, set_seed

def build_pos_vocab(sentences):
    pos_vocab = {}
    for sentence in sentences:
        for word in sentence:   
            pos_tag = word.pos
            if pos_tag not in pos_vocab:
                pos_vocab[pos_tag] = []
            if word.text.lower() not in pos_vocab[pos_tag]:
                pos_vocab[pos_tag].append(word.text.lower())
    return pos_vocab

mask_token = "<mask>"

def make_sample(input_sentence, pos_vocab, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    sentence = []
    for word in input_sentence:
        # apply single token masking or POS-guided replacing
        u = np.random.uniform()
        if u < p_mask:
            sentence.append(mask_token)
        elif u < (p_mask + p_pos):
            same_pos = pos_vocab[word.pos]
            sentence.append(np.random.choice(same_pos))
        else:
            sentence.append(word.text.lower())
    # apply n-gram sampling
    if len(sentence) > 2 and np.random.uniform() < p_ng:
        n = min(np.random.choice(range(1, 5+1)), len(sentence) - 1)
        start = np.random.choice(len(sentence) - n)
        for idx in range(start, start + n):
            sentence[idx] = mask_token
    return sentence

    
def augment_dataset(input_dataset, pos_vocab, n_iter=20, p_mask=0.1, p_pos=0.1, p_ng=0.25, max_ng=5):
    dataset = []
    for p in tqdm(input_dataset, "Generation"):
        samples = [[word.text.lower() for word in p[0]]]
        for _ in range(n_iter):
            new_sample = make_sample(p[0], pos_vocab, p_mask, p_pos, p_ng, max_ng)
            if new_sample not in samples:
                samples.append(new_sample)
        dataset.extend([[s, p[1]] for s in samples])
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    dataset = load_tsv(args.input, conversions=(None, int))
    sentences = [spacy_en(text) for text, _ in tqdm(dataset, desc="Loading dataset")]
    pos_vocab = build_pos_vocab(sentences)
    for p, s in zip(dataset, sentences):
        p[0] = s
    dataset = augment_dataset(dataset, pos_vocab)
    with open(args.output, "w") as f:
        f.write("sentence\tlabel\n")
        for sentence, rating in dataset:
            f.write(" ".join(sentence) + "\t" + str(rating) + "\n")
        
    
