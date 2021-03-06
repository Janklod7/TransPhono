from datetime import time

import yaml
from torch import nn, Tensor
import torch
import math
import time

from torch.nn import TransformerEncoderLayer, TransformerEncoder

from datasets.language_loader import LanguageData


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


class TransPhono(nn.Module):

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

        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout, max_len=int(self.config["batch_size"]))
        encoder_layers = TransformerEncoderLayer(self.embed_size, self.nhead, self.d_hid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        self.decoder = nn.Linear(self.embed_size, self.ntoken)
        if features is not None:
            self.encoder = nn.Embedding.from_pretrained(features, freeze=True)
        else:
            self.encoder = nn.Embedding(self.ntoken, self.embed_size)
            self.init_weights()
        self.sftmx = nn.Softmax(dim=2)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_hid)

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
