import json
import math
from typing import Tuple

import torch
import yaml
import os
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import requests
from datasets.language_loader import Language


class EngLang(Language):
    def __init__(self, root, params, data=None, use_plurals=False, savenew=False):
        with open(root+params) as f:
            self.config = yaml.load(f, Loader=yaml.BaseLoader)['language']
        mx = int(self.config['maximum_word_length'])
        if data and os.path.isfile(f"{root+data}/data_{mx}.json"):
            data = root+data
            with open(f"{data}/data_{mx}.json") as f:
                self.data = json.load(f)
            with open(f"{data}/phonemes_{mx}.json") as f:
                self.phonemes = json.load(f)
            with open(f"{data}/index_to_phon_{mx}.json") as f:
                self.index_to_phon = json.load(f)
            with open(f"{data}/plurals_{mx}.json") as f:
                self.plurals = json.load(f)
        else:
            self.parse_new(savenew)
        if use_plurals:
            self.data.update(self.plurals)

        self.vecs_x, self.vecs_y = self.make_vecs()

    def parse_new(self, save=False):
        response = requests.get("http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b").text.split("\n")
        self.data = {}
        self.index_to_phon = {0: ""}
        self.phonemes = {"": 0}
        self.phonemes["PLUR"] = 1
        toolong = 0
        maxlen = 0
        ip = 2
        for d in response:
            if len(d) == 0 or d[0] == ";":
                continue
            d = d.split(" ")
            phonms = ["BEG"] + d[2:] + ["END"]
            if len(phonms) > int(self.config['maximum_word_length']):
                toolong += 1
                continue
            self.data[d[0]] = (phonms, phonms)
            if len(phonms) > maxlen:
                maxlen = len(phonms)
            for p in phonms:
                if p not in self.phonemes:
                    self.phonemes[p] = ip
                    self.index_to_phon[ip] = p
                    ip += 1
        pa = "datasets/Languages/English"
        with open(f"{pa}/phonemes_{self.config['maximum_word_length']}.json", 'w') as fp:
            json.dump(self.phonemes, fp)
        with open(f"{pa}/data_{self.config['maximum_word_length']}.json", 'w') as fp:
            json.dump(self.data, fp)
        with open(f"{pa}/index_to_phon_{self.config['maximum_word_length']}.json", 'w') as fp:
            json.dump(self.index_to_phon, fp)

        self.plurals = {}
        for w, p in self.data.items():
            if len(w) >= 4 and w[-1] == "S":
                if w[:-1] in self.data:
                    # self.plurals.append((w[:-1], w))
                    self.plurals[w[:-1] + "_PLUR"] = (self.data[w[:-1]][0][:-1] + ["PLUR", "END"], p)

                elif w[:-2] in self.data and w[-2] == "E":
                    self.plurals[w[:-2] + "_PLUR"] = (self.data[w[:-2]][0][:-1] + ["PLUR", "END"], p)

                elif w[-3:] == "IES" and w[:-3] + "Y" in self.data:
                    self.plurals[w[:-3] + "_PLUR"] = (self.data[w[:-3] + "Y"][0][:-1] + ["PLUR", "END"], p)

        with open(f"{pa}/plurals_{self.config['maximum_word_length']}.json", 'w') as fp:
            json.dump(self.plurals, fp)

        # for w in self.plurals:
        #     self.data[w] = self.plurals[w]

        print(f"Got {len(self.data) - len(self.plurals)} words")
        print(f"Got {len(self.phonemes)} phonemes")
        print(f"Got {len(self.plurals)} plurals")
        print(f"Word maximum length is {maxlen}")
        print(f"{toolong} Words were too long")

    def word2vec(self, features):
        return torch.tensor([self.phonemes[a] for a in features])

    def make_vecs(self):
        return torch.nn.utils.rnn.pad_sequence([self.word2vec(w) for w, l in self.data.values()],
                                               batch_first=True), \
               torch.nn.functional.one_hot(
                   torch.nn.utils.rnn.pad_sequence([self.word2vec(l) for w, l in self.data.values()], batch_first=True),
                   num_classes=len(self.phonemes))

    def getX(self):
        return self.vecs_x

    def getY(self):
        return self.vecs_y


if __name__ == "__main__":
    eng = EngLang("english_config.yaml", data="Languages/English")
    print(list(eng.data.values())[:10])
    print(eng.phonemes)
