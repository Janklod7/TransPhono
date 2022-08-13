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
        self.seq_len = int(self.dataset.language.config['maximum_word_length'])
        self.isFeatures = False

        if features is not None:
            self.encoder = nn.Embedding.from_pretrained(features, freeze=True)
            self.learned_encoder = nn.Embedding(self.ntoken, self.embed_size - len(features[0]))
            self.ntoken = len(features[0])
            # self.embed_size = len(features[0])
            self.isFeatures = True
        else:
            self.learned_encoder = nn.Embedding(self.ntoken, self.embed_size)
            self.learned_encoder.weight.data.uniform_(-0.1, 0.1)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout, max_len=self.seq_len)
        encoder_layers = TransformerEncoderLayer(d_model=self.embed_size,
                                                 nhead=self.nhead, dim_feedforward=self.d_hid, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        self.decoder = nn.Linear(self.embed_size, self.ntoken)
        self.sftmx = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, isTan=False) -> Tensor:
        src = torch.permute(src, (1,0))
        return self.decode(self.encode(src, src_mask, isTan), src_mask)

    def embed(self, src):
        if self.isFeatures:
            src_f = self.encoder(src)
            src = self.learned_encoder(src)
            src = torch.concat((src_f, src), dim=-1) * math.sqrt(self.d_hid)
        else:
            src = self.learned_encoder(src) * math.sqrt(self.d_hid)
        return src

    def encode(self, src: Tensor, src_mask: Tensor, isTan=False) -> Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        # src = self.transformer_encoder(src, src_mask)
        if isTan:
            src = self.tanh(src)
        return src

    def decode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.transformer_encoder(src, src_mask)
        return torch.permute(self.decoder(src), (1,0,2))

    def generate_noise(self, batch_size, isTan=False, noise_gen=None):
        noise = (torch.rand((self.seq_len, batch_size, self.embed_size), generator=noise_gen) * 2) - 1
        if isTan:
            noise = self.tanh(noise)
        return noise

    def train_epoch(self, train_data, parameters, gen_optimizer, criterion, epoch, isTan=False):

        batch_size = int(parameters["batch_size"])
        log_interval = int(parameters["log_intervals"])
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, 1, gamma=0.95)
        self.train()
        total_loss = 0.
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(self.seq_len).to(self.device)

        num_batches = len(train_data)
        batch = 0
        for data, targets in iter(train_data):
            if len(data[0]) != self.seq_len:  # only on last batch
                src_mask = src_mask[:data.shape[0], :data.shape[0]]

            data = data.to(self.device)
            output = self(data, src_mask, isTan)
            if self.isFeatures:
                with torch.no_grad():
                    targets = self.encoder(data)
            loss = 10 * criterion(output, targets.float().to(self.device))

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
        src_mask = generate_square_subsequent_mask(self.seq_len).to(self.device)
        with torch.no_grad():
            if show_example > 0:
                example_x, example_y = next(iter(eval_data))
                example_x = example_x.to(self.device)
                # real_words_1 = self.dataset.vec2word(example_y)
                words = []
                dist = torch.nn.L1Loss(reduction='none')
                # a = self(example_x, src_mask)
                # b =
                # words = [np.argmax([torch.sum(dist(p, self.dataset.language.features.feature_mat), dim=1) for p in w]) for w in a]

                for w in self(example_x, src_mask):
                    word = []
                    for p in w:
                        dists = torch.sum(dist(p, self.dataset.language.features.feature_mat), dim=1)
                        best = torch.argmin(dists).item()
                        best = self.dataset.language.index_to_phon[best]
                        if best != "":
                            word.append(best)
                    words.append(" ".join(word))
                # print(words)

                if self.isFeatures:
                    with torch.no_grad():
                        example_y = self.encoder(example_x)
                fake_words = self.dataset.vec2word(self(example_x, src_mask), self.isFeatures)
                real_words = self.dataset.vec2word(example_y, self.isFeatures)

                # latent = self.encode(torch.permute(example_x, (1, 0)), src_mask, False)
                # latent_shape = latent.shape
                # forhist = latent.flatten().to("cpu").numpy()
                # print("Latent space hystogram no TanH: ")
                # plt.hist(forhist, bins=50)
                # plt.show()
                #
                # forhist = self.tanh(latent)
                # forhist = forhist.flatten().to("cpu").numpy()
                # print("Latent space hystogram: ")
                # plt.hist(forhist, bins=50)
                # plt.show()
                # print(latent_shape)
                # # noise = latent[torch.randperm(latent_shape[0]),torch.randperm(latent_shape[1]),torch.randperm(latent_shape[2])]
                # noise = latent[:,:,torch.randperm(latent_shape[2])]
                # # noise = ((torch.randn(latent_shape, device=self.device, generator=torch.Generator(device=self.device)) * 2) - 1)
                # # noise = torch.log(noise) / torch.max(noise).item() * 2
                # # # noise = (torch.ones(noise.shape) * 2.5).to(self.device) - noise
                # forhist = noise.flatten().to("cpu").numpy()
                # print("noise hystogram b4 tanh: ")
                # plt.hist(forhist, bins=50)
                # plt.show()
                # noise = self.tanh(noise)
                # forhist = noise.flatten().to("cpu").numpy()
                # print("noise hystogram after tanh: ")
                # plt.hist(forhist, bins=50)
                # plt.show()
                # noise =torch.permute(self.decode(noise, src_mask), (1,0,2))
                # noises = self.dataset.vec2word(noise)
                # print("NOISES:", )
                # for n in noises[:show_example]:
                #     print(n)
                print("RECONSTRUCTS:")
                for f, r in zip(fake_words[:show_example], real_words[:show_example]):
                    print(f, ",", r)
            for data, targets in iter(eval_data):
                if len(data[0]) != self.seq_len:  # only on last batch
                    src_mask = src_mask[:data.shape[0], :data.shape[0]]
                if self.isFeatures:
                    with torch.no_grad():
                        targets = self.encoder(data)
                data = data.to(self.device)
                output = self(data, src_mask)
                total_loss += batch_size * criterion(output, targets.float().to(self.device)).item()

        return total_loss / ((len(eval_data) * batch_size - 1) - 1)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
