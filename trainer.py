import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchtext import data

from tensorboardX import SummaryWriter

from tqdm.autonotebook import tqdm, trange

class Trainer():
    def __init__(self, model, device,
        loss="cross_entropy",
        train_dataset=None,
        temperature=1.0,
        val_dataset=None, val_interval=1,
        checkpt_callback=None, checkpt_interval=1,
        max_grad_norm=1.0, batch_size=64, gradient_accumulation_steps=1,
        lr=5e-5, weight_decay=0.0):
        # Storing
        self.model = model
        self.device = device
        self.loss_option = loss
        self.train_dataset = train_dataset
        self.temperature = temperature
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.checkpt_callback = checkpt_callback
        self.checkpt_interval = checkpt_interval
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        # Initialization
        assert self.loss_option in ["cross_entropy", "mse", "kl_div"]
        if self.loss_option == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        elif self.loss_option == "mse":
            self.loss_function = nn.MSELoss(reduction="sum")
        elif self.loss_option == "kl_div":
            self.loss_function = nn.KLDivLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.train_dataset is not None:
            self.train_it = data.BucketIterator(self.train_dataset, self.batch_size, train=True, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.train_it = None
        if self.val_dataset is not None:
            self.val_it = data.BucketIterator(self.val_dataset, self.batch_size, train=False, sort_key=lambda x: len(x.text), device=self.device)
        else:
            self.val_it = None
    def get_loss(self, model_output, label, curr_batch_size):
        if self.loss_option in ["cross_entropy", "mse"]:
            loss = self.loss_function(
                model_output,
                label
            ) / curr_batch_size # Mean over batch
        elif self.loss_option == "kl_div":
            # KL Divergence loss needs special care
            # It expects log probabilities for the model's output, and probabilities for the label
            loss = self.loss_function(
                F.log_softmax(model_output / self.temperature, dim=-1),
                F.softmax(label / self.temperature, dim=-1)
            ) / (self.temperature * self.temperature) / curr_batch_size
        return loss
    def train_step(self, batch):
        self.model.train()
        batch, label, curr_batch_size = self.process_batch(batch)
        s_logits = self.model(**batch)[0]
        loss = self.get_loss(s_logits, label, curr_batch_size)
        loss.backward()
        self.training_step += 1
        if self.training_step % self.gradient_accumulation_steps == 0:
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
            self.tb_writer.add_scalar("loss", loss, self.global_step)
            self.global_step += 1
            # Every val_interval steps, evaluate and log stats to tensorboard
            if self.val_interval >= 0 and (self.global_step + 1) % self.val_interval == 0:
                results = self.evaluate()
                print(results)
                for k, v in results.items():
                    self.tb_writer.add_scalar("val_" + k, v, self.global_step)
            # Every checkpt_interval steps, call checkpt_callback to save a checkpoint
            if self.checkpt_interval >= 0 and (self.global_step + 1) % self.checkpt_interval == 0:
                self.checkpt_callback(self.model, self.global_step)
    def train(self, epochs=1, schedule=None, **kwargs):
        # Initialization
        self.global_step = 0
        self.training_step = 0
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
            for batch in tqdm(self.train_it, desc="Epoch %d" % epoch):
                self.train_step(batch)
        self.tb_writer.close()
        del self.tb_writer
    def evaluate(self):
        self.model.eval()
        val_loss = val_accuracy = 0.0
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        for batch in tqdm(self.val_it, desc="Evaluation"):
            with torch.no_grad():
                batch, label, _ = self.process_batch(batch)
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
    def infer(self, dataset, softmax=False):
        self.model.eval()
        outputs_idx = 0
        outputs = np.empty(shape=(len(dataset), 2))
        infer_it = data.Iterator(dataset, self.batch_size, train=False, sort=False, device=self.device)
        for batch in tqdm(infer_it, desc="Inference"):
            with torch.no_grad():
                batch, _, batch_size = self.process_batch(batch)
                output = self.model(**batch)[0]
                if softmax:
                    output = F.softmax(output, dim=-1)
                outputs[outputs_idx:outputs_idx + batch_size] = output.detach().cpu().numpy()
                outputs_idx += batch_size
                del output
        return outputs
    def infer_one(self, example, text_field=None, softmax=False):
        self.model.eval()
        if text_field is None:
            text_field = self.train_dataset.fields["text"]
        example = text_field.preprocess(example)
        tokens, length  = text_field.process([example])
        with torch.no_grad():
            batch = self.process_one(tokens, length)
            output = self.model(**batch)[0]
            if softmax:
                output = F.softmax(output, dim=-1)
            output = output.detach().cpu().numpy()
        return output[0]
    def process_batch(self, *args):
        # Implemented by subclasses
        raise NotImplementedError()
    def process_one(self, *args):
        # Implemented by subclasses
        raise NotImplementedError()

class BertTrainer(Trainer):
    def process_batch(self, batch):
        tokens, length = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        length = length.unsqueeze_(1).expand(tokens.size())
        rg = torch.arange(tokens.size(1), device=self.device).unsqueeze_(0).expand(tokens.size())
        attention_mask = (rg < length).type(torch.float32)
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        return batch, label, tokens.size(0)
    def process_one(self, tokens, length):
        return {
            "input_ids": tokens.to(self.device),
            "attention_mask": torch.ones(tokens.size(), dtype=torch.float32, device=self.device)
        }

class LSTMTrainer(Trainer):
    def process_batch(self, batch):
        tokens, length = batch.text
        label = batch.label if "label" in batch.__dict__ else None
        batch = {
            "seq": tokens,
            "length": length,
        }
        return batch, label, tokens.size(1)
    def process_one(self, tokens, length):
        return {
            "seq": tokens.to(self.device),
            "length": length.to(self.device)
        }
