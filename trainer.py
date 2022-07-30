import time
import yaml
import torch
import math
import copy
from torch import nn

from datasets.language_loader import LanguageData
from models.basic_transformer import TransPhono, generate_square_subsequent_mask

model_map = {"basic transformer": TransPhono,
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


    def train_reconstruct(self, parameters=None):
        if not parameters:
            parameters = self.config
        generator = model_map[parameters["model_type"]](self.dataset, parameters, device=self.device) #TODO use features
        criterion = nn.CrossEntropyLoss()
        batch_size = int(parameters["batch_size"])
        train_data, test_data = self.dataset.generate_loader(batch_size, float(parameters["train_size"]), device=self.device)
        example_x, example_y = next(iter(test_data))
        print(f"size of embedding:", generator.encoder(example_x).shape)

        gen_optimizer = torch.optim.SGD(generator.parameters(), lr=self.gen_lr)
        best_val_loss = float('inf')

        best_model = None
        for epoch in range(1, int(parameters["epoches"]) + 1):
            epoch_start_time = time.time()
            generator.train_epoch(train_data, parameters, gen_optimizer, criterion, epoch)
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



