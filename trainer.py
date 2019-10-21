import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchtext import data

from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

class Trainer():
    def __init__(self, model, train_dataset, fasttext_pad_token, bert_pad_token, device,
        teacher=None, teacher_alpha=0.0, teacher_loss = "mse",
        val_dataset=None, val_interval=1,
        checkpt_callback=None, checkpt_interval=1,
        gradient_accumulation_steps=1, max_grad_norm=1.0,
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
        self.checkpt_callback = checkpt_callback
        self.checkpt_interval = checkpt_interval
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
        else:
            t_loss = 0.0
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
        if self.val_dataset is not None and (self.global_step + 1) % self.val_interval == 0:
            results = self.evaluate()
            print(results)
            for k, v in results.items():
                self.tb_writer.add_scalar("val_" + k, v, self.global_step)
        if self.checkpt_callback is not None and (self.global_step + 1) % self.checkpt_interval == 0:
            self.checkpt_callback(self.model, self.global_step)
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
            model = self.teacher
        else:
            model = self.model
        model.eval()
        val_loss = val_accuracy = 0.0
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        data_it = iter(self.val_dataloader)
        data_it = tqdm(data_it, desc="Evaluation", total=len(self.val_dataset) // self.batch_size)
        for batch in data_it:
            fasttext_tokens, _, labels, length, _ = self.process_batch(batch)
            with torch.no_grad():
                    output = model(fasttext_tokens, length)
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