# distil-bilstm

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)

This repository contains scripts to train a tiny bidirectional LSTM classifier on the SST-2 dataset (url).
It also contains a script to fine-tune `bert-large-uncased` on the same task.
The procedure is inspired by the paper [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136).

### Installing requirements

```bash
pip install -r requirements.txt  # Skip this if you are running on FloydHub
python -m spacy download en
```

### Fine-tuning bert-large-uncased


```bash
>> python train_bert.py --help

usage: train_bert.py [-h] --data_dir DATA_DIR --output_dir OUTPUT_DIR
                     [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                     [--lr_schedule {constant,warmup,cyclic}]
                     [--warmup_steps WARMUP_STEPS]
                     [--epochs_per_cycle EPOCHS_PER_CYCLE] [--do_train]
                     [--seed SEED] [--no_cuda] [--cache_dir CACHE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the dataset.
  --output_dir OUTPUT_DIR
                        Directory where to save the model.
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR               Learning rate.
  --lr_schedule {constant,warmup,cyclic}
                        Schedule to use for the learning rate. Choices are:
                        constant, linear warmup & decay, cyclic.
  --warmup_steps WARMUP_STEPS
                        Warmup steps for the 'warmup' learning rate schedule.
                        Ignored otherwise.
  --epochs_per_cycle EPOCHS_PER_CYCLE
                        Epochs per cycle for the 'cyclic' learning rate
                        schedule. Ignored otherwise.
  --do_train
  --seed SEED           Random seed.
  --no_cuda
  --cache_dir CACHE_DIR
                        Custom cache for transformer models.
```

Example:

```bash
python train_bert.py --data_dir SST-2 --output_dir bert_output --epochs 1 --batch_size 16 --lr 1e-5 --lr_schedule warmup --warmup_steps 100 --do_train
```

### Generating the augmented dataset

The file used in my tests is available at https://www.floydhub.com/alexamadori/datasets/sst-2-augmented/1, but you may want to generate another one with a random seed or to use a different teacher model.

```bash
>> python generate_dataset.py --help

usage: generate_dataset.py [-h] --input INPUT --output OUTPUT --model MODEL
                           [--no_augment] [--batch_size BATCH_SIZE]
                           [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input dataset.
  --output OUTPUT       Output dataset.
  --model MODEL         Model to use to generate the labels for the augmented
                        dataset.
  --no_augment          Don't perform data augmentation
  --batch_size BATCH_SIZE
  --no_cuda

```

Example:

```bash
python generate_dataset.py --input SST-2/train.tsv --output SST-2/augmented.tsv --model bert_output
```

### Training the BiLSTM model

```bash
>> python train_bilstm.py --help

usage: train_bilstm.py [-h] --data_dir DATA_DIR --output_dir OUTPUT_DIR
                       [--augmented] [--epochs EPOCHS]
                       [--batch_size BATCH_SIZE] [--lr LR]
                       [--lr_schedule {constant,warmup,cyclic}]
                       [--warmup_steps WARMUP_STEPS]
                       [--epochs_per_cycle EPOCHS_PER_CYCLE] [--do_train]
                       [--seed SEED] [--no_cuda]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the dataset.
  --output_dir OUTPUT_DIR
                        Directory where to save the model.
  --augmented           Wether to use the augmented dataset for knowledge
                        distillation
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR               Learning rate.
  --lr_schedule {constant,warmup,cyclic}
                        Schedule to use for the learning rate. Choices are:
                        constant, linear warmup & decay, cyclic.
  --warmup_steps WARMUP_STEPS
                        Warmup steps for the 'warmup' learning rate schedule.
                        Ignored otherwise.
  --epochs_per_cycle EPOCHS_PER_CYCLE
                        Epochs per cycle for the 'cyclic' learning rate
                        schedule. Ignored otherwise.
  --do_train
  --seed SEED
  --no_cuda
```

Example:

```bash
python train_bilstm.py --data_dir SST-2 --output_dir bilstm_output --epochs 1 --batch_size 50 --lr 1e-3 --lr_schedule warmup --warmup_steps 100 --do_train --augmented
```
