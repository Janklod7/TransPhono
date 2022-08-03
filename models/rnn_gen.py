from datetime import time

import yaml
from torch import nn, Tensor
import torch
import math
import time
from collections import OrderedDict

from datasets.language_loader import LanguageData


class Rnn_Gen(nn.Module):

    def __init__(self, dataset: LanguageData, parameters: dict = None, features=None, device=None):
        super().__init__()
        self.config = parameters
        self.device = device
        self.dataset = dataset
        self.ntoken = len(dataset.language.phonemes)
        self.nfeatures = len(dataset.language.phonemes)
        self.nhead = int(self.config['nhead'])
        self.e_hid = list(map(int, self.config['e_hid'].split(",")))
        self.d_hid = list(map(int, self.config['e_hid'].split(",")))
        self.embed_size = int(self.config['embedding_size'])
        self.dropout = float(self.config['dropout'])

        if features is not None:
            self.nfeatures = len(features[0])
        self.sftmx = nn.Softmax(dim=2)

        self.elayers = []
        last_size = self.nfeatures
        for i, s in enumerate(self.e_hid):
            self.elayers.append(("Linear" + str(i), nn.Linear(last_size, s)))
            self.elayers.append(("ReLU" + str(i), nn.ReLU()))
            last_size = s
        self.elayers.append(nn.Linear(last_size, self.embed_size))

        self.dlayers = []
        last_size = self.embed_size
        for i, s in enumerate(self.d_hid):
            self.dlayers.append(("Linear" + str(i), nn.Linear(last_size, s)))
            self.dlayers.append(("ReLU" + str(i), nn.ReLU()))
            last_size = s
        self.dlayers.append(nn.Linear(last_size, self.ntoken))

        self.encoder = nn.Sequential(OrderedDict(self.elayers))
        self.decoder = nn.Sequential(OrderedDict(self.dlayers))

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.ecoder.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = torch.zeros()
        for p in

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)

        return output

    def generate_words(self, src_mask):

        fake_gen = (torch.rand((self.batch_size, len(self.dataset.language.phonemes), self.embed_size),
                               device=self.device) * 2 - 1) * math.sqrt(self.d_hid)
        src = self.pos_encoder(fake_gen)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)

        return output

    def train_epoch(self, train_data, parameters, gen_optimizer, criterion, epoch):

        sft = nn.Softmax(dim=2)
        batch_size = int(parameters["batch_size"])
        log_interval = int(parameters["log_intervals"])
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 1, gamma=0.95)
        self.train()
        total_loss = 0.

        start_time = time.time()
        src_mask = generate_square_subsequent_mask(batch_size).to(self.device)

        num_batches = len(train_data)
        batch = 0
        for data, targets in iter(train_data):
            if len(data) != batch_size:  # only on last batch
                src_mask = src_mask[:data.shape[0], :data.shape[0]]
            output = self(data, src_mask)
            loss = 10 * criterion(output, targets.float())

            gen_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            gen_optimizer.step()

            total_loss += loss.item()
            batch += 1
            if batch % log_interval == 0 and batch > 0:
                lr = gen_scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()
        gen_scheduler.step()

    def evaluate(self, eval_data: Tensor, parameters, criterion, show_example=0) -> float:
        self.eval()
        batch_size = int(parameters["batch_size"])
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(batch_size).to(self.device)
        with torch.no_grad():
            fake_words = real_words = []
            if show_example > 0:
                example_x, example_y = next(iter(eval_data))
                fake_words = self.dataset.vec2word(self(example_x, src_mask))
                real_words = self.dataset.vec2word(example_y)
            for data, targets in iter(eval_data):
                if len(data) != batch_size:  # only on last batch
                    src_mask = src_mask[:data.shape[0], :data.shape[0]]
                output = self(data, src_mask)
                total_loss += batch_size * criterion(output, targets.float()).item()
            for f, r in zip(fake_words[:show_example], real_words[:show_example]):
                print(f, ",", r)
        return total_loss / ((len(eval_data) * batch_size - 1) - 1)



def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
