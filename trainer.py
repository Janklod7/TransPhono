import time
import yaml
import torch
import math
import copy
from torch import nn

from datasets.language_loader import LanguageData
from models.basic_transformer import TransPhono
from models.basic_transformer_disc import TransPhonoDisc

model_map = {"basic transformer": TransPhono,
             "basic transformer disc": TransPhonoDisc
             }


class Trainer:
    def __init__(self, dataset: LanguageData, parameters):
        with open(parameters) as f:
            self.config = yaml.load(f, Loader=yaml.BaseLoader)
        self.dataset = dataset
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.gen_lr = float(self.config['generator_lr'])
        self.disc_lr = float(self.config['discriminator_lr'])
        self.epoches = int(self.config['epoches'])
        self.log_intervals = int(self.config['log_intervals'])


    def train_reconstruct(self, parameters=None, isTan=False):
        if not parameters:
            parameters = self.config
        generator = model_map[parameters["gen_type"]](self.dataset, parameters, device=self.device).to(self.device) #TODO use features
        criterion = nn.CrossEntropyLoss()
        batch_size = int(parameters["batch_size"])
        train_data, test_data = self.dataset.generate_loader(batch_size, float(parameters["train_size"]), device=self.device)
        example_x, example_y = next(iter(test_data))
        print(f"size of embedding:", generator.encoder(example_x.to(self.device)).shape)

        gen_optimizer = torch.optim.SGD(generator.parameters(), lr=self.gen_lr)
        best_val_loss = float('inf')

        best_model = None
        for epoch in range(1, int(parameters["epoches"]) + 1):
            epoch_start_time = time.time()
            generator.train_epoch(train_data, parameters, gen_optimizer, criterion, epoch, isTan)
            val_loss = 10 * generator.evaluate(test_data, parameters, criterion, 5)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(generator)
        return best_model

    def train_gan(self, parameters=None, isTan=False):
        if not parameters:
            parameters = self.config
        generator = model_map[parameters["gen_type"]](self.dataset, parameters, device=self.device).to(self.device) #TODO use features
        discriminator = model_map[parameters["disc_type"]](self.dataset, parameters, device=self.device).to(
            self.device)  # TODO use features
        criterion = nn.CrossEntropyLoss()
        batch_size = int(parameters["batch_size"])
        train_data, test_data = self.dataset.generate_loader(batch_size, float(parameters["train_size"]), device=self.device)
        example_x, example_y = next(iter(test_data))
        print(f"size of embedding:", generator.encoder(example_x.to(self.device)).shape)

        gen_optimizer = torch.optim.SGD(generator.parameters(), lr=self.gen_lr)
        disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr=self.disc_lr)
        best_val_loss = float('inf')

        best_gan = None
        best_disc = None
        for epoch in range(1, int(parameters["epoches"]) + 1):
            epoch_start_time = time.time()
            discriminator.train_epoch(train_data, parameters, generator, gen_optimizer, disc_optimizer, criterion, epoch, isTan)
            disc_loss, gen_loss = discriminator.evaluate(test_data, generator, parameters, criterion, 5)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'discriminator loss {disc_loss:5.2f} | generator loss {gen_loss:8.2f}')
            print('-' * 89)


            if disc_loss + gen_loss < best_val_loss:
                best_val_loss = disc_loss + gen_loss
                best_gan = copy.deepcopy(generator)
                best_disc = copy.deepcopy(discriminator)
        return best_gan, best_disc



