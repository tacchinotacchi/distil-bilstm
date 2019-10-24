import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchtext import data

from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

class Trainer():
    def __init__(self, model, loss, pad_token, device,
        train_dataset=None,
        temperature=1.0,
        val_dataset=None, val_interval=1,
        checkpt_callback=None, checkpt_interval=1,
        gradient_accumulation_steps=1, max_grad_norm=1.0,
        batch_size=50, lr=5e-5, weight_decay=0.0):
        # Storing
        self.model = model
        self.train_dataset = train_dataset
        self.pad_token = pad_token
        self.device = device
        self.loss = loss
        self.temperature = temperature
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.checkpt_callback = checkpt_callback
        self.checkpt_interval = checkpt_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        # Initialization
        assert self.loss in ["cross_entropy", "mse", "kl_div"]
        if self.loss == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        elif self.loss == "mse":
            self.loss_function = nn.MSELoss(reduction="sum")
        elif self.loss == "kl_div":
            self.loss_function = nn.KLDivLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.train_dataset is not None:
            self.train_dataloader = data.Iterator(self.train_dataset, self.batch_size, train=True, shuffle=True, device=self.device)
        else:
            self.train_dataloader = None
        if self.val_dataset is not None:
            self.val_dataloader = data.Iterator(self.val_dataset, self.batch_size, train=False, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.val_dataloader = None
    def get_loss(self, model_output, label, curr_batch_size):
        if self.loss in ["cross_entropy", "mse"]:
            loss = self.loss_function(
                model_output,
                label
            ) / curr_batch_size # Mean over batch
        elif self.loss == "kl_div":
            # KL Divergence loss needs special care
            # It expects log probabilities for the model's output, and probabilities for the label
            loss = self.loss_function(
                F.log_softmax(model_output, dim=-1) / self.temperature,
                F.softmax(label, dim=-1) / self.temperature
            ) / (self.temperature * self.temperature) / curr_batch_size
        return loss
    def train_step(self, batch):
        self.model.train()
        batch, label, curr_batch_size = self.process_batch(batch)
        s_logits = self.model(**batch)[0]
        loss = self.get_loss(s_logits, label, curr_batch_size)
        loss.backward()
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            # Advance learning rate schedule
            self.scheduler.step()
        self.model.zero_grad()
        # Save stats to tensorboard
        self.tb_writer.add_scalar("lr",
            self.scheduler.get_lr()[0] if self.scheduler is not None else self.lr,
            self.global_step)
        self.tb_writer.add_scalar("loss", self.tb_loss, self.global_step)
        self.global_step += 1
        # Every val_interval steps, evaluate and log stats to tensorboard
        if self.val_dataset is not None and (self.global_step + 1) % self.val_interval == 0:
            results = self.evaluate()
            print(results)
            for k, v in results.items():
                self.tb_writer.add_scalar("val_" + k, v, self.global_step)
        # Every checkpt_interval steps, call checkpt_callback to save a checkpoint
        if self.checkpt_callback is not None and (self.global_step + 1) % self.checkpt_interval == 0:
            self.checkpt_callback(self.model, self.global_step)
    def train(self, epochs=1, schedule=None, **kwargs):
        # Initialization
        self.global_step = 0
        self.tb_writer = SummaryWriter()
        steps_per_epoch = len(self.train_dataset) // self.batch_size // self.gradient_accumulation_steps
        total_steps = epochs * steps_per_epoch
        # Initialize the learning rate scheduler if one has been chosen
        assert schedule is None or schedule in ["warmup", "cyclic"]
        if schedule is None:
            self.scheduler = None
            for grp in self.optimizer.param_groups: grp['lr'] = self.lr
        elif schedule == "warmup":
            warmup_steps = kwargs["warmup_steps"]
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/100, max_lr=self.lr,
                step_size_up=max(1, warmup_steps), step_size_down=(total_steps - warmup_steps),
                cycle_momentum=False)
        elif schedule == "cyclic":
            epochs_per_cycle = kwargs["epochs_per_cycle"]
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/25, max_lr=self.lr,
            step_size_up=steps_per_epoch // 2,
            cycle_momentum=False)
        # The loop over epochs, batches
        for epoch in trange(epochs, desc="Training"):
            data_it = iter(self.train_dataloader)
            for batch in tqdm(data_it, desc="Epoch %d" % epoch, total=len(self.train_dataloader)):
                self.train_step(batch)
        self.tb_writer.close()
        del self.tb_writer
    def evaluate(self):
        self.model.eval()
        val_loss = val_accuracy = 0.0
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        data_it = iter(self.val_dataloader)
        for batch in tqdm(data_it, desc="Evaluation", total=len(self.val_dataloader)):
            batch, label, _ = self.process_batch(batch)
            with torch.no_grad():
                output = self.model(**batch)[0]
                loss = loss_func(output, label)
                val_loss += loss.item()
                val_accuracy += (output.argmax(dim=-1) == label).sum().item()
        val_loss /= len(self.val_dataset)
        val_accuracy /= len(self.val_dataset)
        return {
            "loss": val_loss,
            "perplexity": np.exp(val_loss),
            "accuracy": val_accuracy
        }
    def infer(self, dataset):
        self.model.eval()
        outputs_idx = 0
        outputs = np.empty(shape=(len(dataset), 2))
        dataloader = data.Iterator(dataset, self.batch_size, train=False, sort=False, device=self.device)
        data_it = iter(dataloader)
        for batch in tqdm(data_it, desc="Inference", total=len(dataloader)):
            batch, _, batch_size = self.process_batch(batch)
            with torch.no_grad():
                output = self.model(**batch)[0]
                outputs[outputs_idx:outputs_idx + batch_size] = output.detach().cpu().numpy()
                outputs_idx += batch_size
                del output
        return np.concatenate(outputs, 0)
    def process_batch(self):
        # Implemented by subclasses
        raise NotImplementedError()

class BertTrainer(Trainer):
    def process_batch(self, batch):
        tokens = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        attention_mask = torch.empty(tokens.size()).to(tokens.device)
        for idx in range(tokens.size(0)):
            attention_mask[idx] = (tokens[idx] != self.pad_token).type(attention_mask.type())
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        return batch, label, tokens.size(0)

class LSTMTrainer(Trainer):
    def process_batch(self, batch):
        tokens = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        length = torch.empty(tokens.size(1), dtype=torch.long).to(tokens.device)
        for idx in range(tokens.size(1)):
            rg = torch.arange(tokens.size(0), device=self.device)
            mask = (tokens[:, idx] != self.pad_token).type(torch.long)
            length[idx] = (rg * mask).argmax() + 1
        batch = {
            "seq": tokens,
            "length": length,
        }
        return batch, label, tokens.size(1)
