from datetime import time

import yaml
from torch import nn, Tensor
import torch
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import TransformerEncoderLayer, TransformerEncoder

from datasets.language_loader import LanguageData
from models.basic_transformer import TransPhono


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransPhonoDisc(nn.Module):

    def __init__(self, dataset: LanguageData, parameters: dict = None, features=None, device=None):
        super().__init__()
        self.config = parameters
        self.device = device
        self.dataset = dataset
        self.ntoken = len(dataset.language.phonemes)
        self.features = len(dataset.language.phonemes)
        self.nhead = int(self.config['nhead'])
        self.d_hid = int(self.config['d_hid'])
        self.nlayers = int(self.config['nlayers'])
        self.embed_size = int(self.config['embedding_size'])
        self.dropout = float(self.config['dropout'])
        self.seq_len = int(self.dataset.language.config['maximum_word_length'])

        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout, max_len=self.seq_len)
        encoder_layers = TransformerEncoderLayer(d_model=self.embed_size,
            nhead=self.nhead, dim_feedforward=self.d_hid, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        self.decoder = nn.Linear(self.embed_size, 1)
        if features is not None:
            self.encoder = nn.Embedding.from_pretrained(features, freeze=True)
        else:
            self.encoder = nn.Embedding(self.ntoken, self.embed_size)
            self.init_weights()
        self.sftmx = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = torch.permute(src, (1,0))
        src = self.encoder(src) * math.sqrt(self.d_hid)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask)
        src = self.decoder(src)
        return torch.permute(src, (1, 0, 2))

    def train_epoch(self, train_data, parameters, generator: TransPhono, gen_optimizer, disc_optimizer, criterion, epoch, isTan=False):

        log_interval = int(parameters["log_intervals"])
        batch_size = int(self.config["batch_size"])
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 1, gamma=0.95)
        disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, 1, gamma=0.95)
        self.train()
        generator.train()
        total_loss = {"real": 0., "fake": 0., "reconstruct": 0., "noise": 0.}
        noise_gen = torch.Generator()

        ones_target = torch.ones((batch_size, 1)).long().to(self.device)
        zeros_target = torch.zeros((batch_size, 1)).long().to(self.device)
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(self.seq_len).to(self.device)
        num_batches = len(train_data)
        batch = 0

        for data, targets in iter(train_data):
            if len(data[0]) != self.seq_len:  # only on last batch
                src_mask = src_mask[:data.shape[0], :data.shape[0]]
            sgmd = nn.Sigmoid()
            data = data.to(self.device)

            disc_optimizer.zero_grad()
            real_disc = self(data, src_mask)
            real_loss = criterion(real_disc, ones_target)
            total_loss["real"] += real_loss.item()

            noise = generator.generate_noise(batch_size, isTan, noise_gen)
            fakes_disc = generator.decode(noise, src_mask)
            fakes_disc = self(torch.max(sgmd(fakes_disc), dim=2).indices, src_mask)
            fake_loss = criterion(fakes_disc, zeros_target)
            total_loss["fake"] += fake_loss.item()

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            disc_optimizer.step()

            gen_optimizer.zero_grad()
            output = generator(data, src_mask, isTan)
            reconstruct_loss = 4*criterion(output, targets.float().to(self.device))
            total_loss["reconstruct"] += reconstruct_loss.item()

            fake_gen = generator.decode(noise, src_mask)
            fake_gen = self(torch.max(sgmd(fake_gen), dim=2).indices, src_mask)
            noise_loss = criterion(fake_gen, ones_target)
            total_loss["noise"] += noise_loss.item()

            gen_loss = reconstruct_loss + noise_loss
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
            gen_optimizer.step()

            batch += 1
            if batch % log_interval == 0 and batch > 0:
                lr = gen_scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval

                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'real_loss {total_loss["real"]/log_interval:5.2f} |',
                      f'fake_loss {total_loss["fake"]/log_interval:5.2f} |',
                      f'reconstruct_loss {total_loss["reconstruct"]/log_interval:5.2f} |',
                      f'noise_loss {total_loss["noise"]/log_interval:5.2f} |',)
                total_loss = {"real": 0., "fake": 0., "reconstruct": 0., "noise": 0.}
                start_time = time.time()
        gen_scheduler.step()
        disc_scheduler.step()

    def evaluate(self, eval_data: Tensor, generator, parameters, criterion, show_example=0, isTan= False):
        batch_size = int(self.config["batch_size"])
        self.eval()
        generator.eval()
        total_loss = {"real": 0., "fake": 0., "reconstruct": 0., "noise": 0.}
        noise_gen = torch.Generator()
        example_x, example_y = next(iter(eval_data))
        example_x = example_x.to(self.device)
        ones_target = torch.ones((batch_size, 1)).long().to(self.device)
        zeros_target = torch.zeros((batch_size, 1)).long().to(self.device)
        src_mask = generate_square_subsequent_mask(self.seq_len).to(self.device)
        for data, targets in iter(eval_data):
            if len(data[0]) != self.seq_len:  # only on last batch
                src_mask = src_mask[:data.shape[0], :data.shape[0]]
            sgmd = nn.Sigmoid()
            data = data.to(self.device)

            real_disc = sgmd(self(data, src_mask))
            real_loss = criterion(real_disc, ones_target)
            total_loss["real"] += real_loss.item()

            noise = generator.generate_noise(batch_size, isTan, noise_gen)
            fakes_disc = generator.decode(noise, src_mask)
            fakes_disc = self(torch.max(sgmd(fakes_disc), dim=2).indices, src_mask)
            fake_loss = criterion(fakes_disc, zeros_target)
            total_loss["fake"] += fake_loss.item()

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

            output = generator(data, src_mask, isTan= isTan)
            reconstruct_loss = 4 * criterion(output, targets.float().to(self.device))
            total_loss["reconstruct"] += reconstruct_loss.item()

            fake_gen = generator.decode(noise, src_mask)
            fake_gen = self(torch.max(sgmd(fake_gen), dim=2).indices, src_mask)
            noise_loss = criterion(fake_gen, ones_target)
            total_loss["noise"] += noise_loss.item()

            gen_loss = reconstruct_loss + noise_loss
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)

            fake_words = self.dataset.vec2word(generator(example_x, src_mask))
            real_words = self.dataset.vec2word(example_y)
            print("RECONSTRUCTION:")
            for f, r in zip(fake_words[:show_example], real_words[:show_example]):
                print(f, ",", r)
            noise = generator.generate_noise(batch_size, isTan, noise_gen)
            fake_words = self.dataset.vec2word(generator.decode(noise, src_mask))
            print("GENERATION:")
            for f in fake_words[:show_example]:
                print(f)

            return ((total_loss["real"] + total_loss["fake"]) / ((len(eval_data) * batch_size - 1) - 1), (total_loss["reconstruct"] + total_loss["noise"]) / ((len(eval_data) * batch_size - 1) - 1))


def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
