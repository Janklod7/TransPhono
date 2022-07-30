from abc import ABC, abstractmethod
import yaml

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader, random_split


class Language(ABC):

    def __init__(self):
        self.phonemes = None
        self.data = None
        self.index_to_phon = None

    @abstractmethod
    def getX(self):
        pass

    @abstractmethod
    def getY(self):
        pass


class LanguageData(Dataset):
    def __init__(self, language_class: Language, params):
        self.language = language_class
        with open(params) as f:
            self.config = yaml.load(f, Loader=yaml.BaseLoader)["dataset"]
        self.x = self.language.getX()
        self.y = self.language.getY()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

    def generate_loader(self, batch_size: int, size, device):
        length = len(self.x)
        if not size:
            size = float(self.config["train_size"])
        train_size = int(length * size)
        test_size = length - train_size
        training_data, test_data = random_split(self, [train_size, test_size])
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

    def vec2word(self, output):
        phoneme_indices = torch.max(output, dim=2).indices
        words = []
        for w in phoneme_indices:
            word = []
            for p in w:
                if p > 0:
                    word.append(self.language.index_to_phon[p.item()])
            word = " ".join(word)
            words.append(word)
        return words





