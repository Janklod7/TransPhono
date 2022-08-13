import torch
from scipy import spatial
import torch.nn.functional as F
class EngFeatures:

    def __init__(self, pad=(0,0)):
        self.names = "voc, cons, high, back, low, ant, cor, round, voice, cont, nasal, strid, edge".split(", ")
        self.ipa = {'': '', 'AA': 'ɑ', 'AE': 'æ', 'AH0': 'ə', 'AH': 'ʌ',  'AO': 'ɔ',
                    'EH': 'ɛ', 'IH': 'ɪ', 'IY': 'i', 'UH': 'ʊ', 'UW': 'u', 'EE': 'e', 'OO': 'o',
                    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'g', 'HH': 'h', 'JH': 'dʒ',
                    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'r', 'S': 's',
                    'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
                    'BEG': '[ ', 'END': ' ]'}
        self.feature_dict = {'': [0,0,0,0,0,0,0,0,0,0,0,0,0],
                             'AA': [1,-1,-1,1,1,-1,-1,-1,0,0,0,0,0],
                             'AE': [1,-1,-1,-1,1,-1,-1,-1,0,0,0,0,0],
                             'AH0': [-1,0,0,0,0,0,0,0,0,0,0,0,0],
                             'AH': [1,-1,-1,1,-1,-1,-1,-1,0,0,0,0,0],
                             'AO': [1,-1,-1,1,1,-1,-1,1,0,0,0,0,0],
                             'EH': [-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0],
                             'IH': [1,-1,1,0,-1,-1,-1,-1,0,0,0,0,0],
                             'IY': [1,-1,1,-1,-1,-1,-1,-1,0,0,0,0,0],
                             'UW': [1,-1,1,1,-1,-1,-1,1,0,0,0,0,0],
                             'UH': [1,-1,0,1,-1,-1,-1,1,0,0,0,0,0],
                             'EE': [1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0],
                             'OO': [1,-1,-1,1,-1,-1,-1,1,0,0,0,0,0],
                             'B': [-1,1,-1,-1,-1,1,-1,0,1,-1,-1,-1,0],
                             'CH': [-1,1,-1,-1,-1,-1,1,0,-1,1,-1,1,0],
                             'D': [-1,1,-1,-1,-1,1,1,0,1,-1,-1,-1,0],
                             'DH': [-1,1,-1,-1,-1,1,1,0,1,1,-1,-1,0],
                             'F': [-1,1,-1,-1,-1,1,-1,0,-1,1,-1,1,0],
                             'G': [-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,0],
                             'HH': [-1,-1,-1,-1,1,-1,-1,0,-1,1,-1,-1,0],
                             'JH': [-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,0],
                             'ZH': [-1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,0],
                             'SH': [-1,1,-1,-1,-1,-1,1,0,1,1,-1,1,0],
                             'K': [-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,0],
                             'L': [1,1,-1,-1,-1,1,1,0,1,1,-1,-1,0],
                             'M': [-1,1,-1,-1,-1,1,-1,0,1,-1,1,-1,0],
                             'N': [-1,1,-1,-1,-1,1,1,0,1,-1,1,-1,0],
                             'NG': [-1,1,1,1,-1,-1,-1,0,1,-1,1,-1,0],
                             'P': [-1,1,-1,-1,-1,1,-1,0,-1,-1,-1,-1,0],
                             'R': [1,1,-1,-1,-1,-1,1,0,1,1,-1,-1,0],
                             'S': [-1,1,-1,-1,-1,1,1,0,-1,1,-1,1,0],
                             'T': [-1,1,-1,-1,-1,1,1,0,-1,-1,-1,-1,0],
                             'TH': [-1,1,-1,-1,-1,1,1,0,-1,1,-1,-1,0],
                             'V': [-1,1,-1,-1,-1,1,-1,0,1,1,-1,1,0],
                             'W': [-1,-1,1,1,-1,-1,-1,1,0,0,0,0,0],
                             'Y': [-1,-1,1,-1,-1,-1,-1,-1,0,0,0,0,0],
                             'Z': [-1,1,-1,-1,-1,1,1,0,1,1,-1,1,0],
                             'BEG': [0,0,0,0,0,0,0,0,0,0,0,0,-1],
                             'END': [0,0,0,0,0,0,0,0,0,0,0,0,1]}
        self.phon_lst = list(self.feature_dict.keys())
        self.feature_mat = torch.tensor(list(self.feature_dict.values())).float()
        self.feature_mat = F.pad(self.feature_mat, pad, "constant", 0)
        # OY = AO, IH. OW = OO, AW. EY = EE, IH. AI = AH, IH. ER = AH0, R. EE = e, OO = o
        self.replace = {'OY': ['AO', 'IH'], 'OW': ['OO', 'UH'], 'AW': ['AH', 'UH'],'AY': ['AH', 'IH'],
                        'EY': ['EE', 'IH'], 'ER': ['AH0', 'R']}
        self.num_of_features = 13
        self.tree = spatial.KDTree(self.feature_mat.detach().numpy())


if __name__ == "__main__":
    eng = EngFeatures(pad=(0,1))
    print(eng.names)
    for p in eng.feature_dict:
        if len(eng.feature_dict[p]) != 13:
            print(p, len(eng.feature_dict[p]))
    print(eng.feature_mat)
    print(eng.feature_mat.shape)
    for p in eng.ipa:
        if p not in eng.feature_dict:
            print(p)

    loss = torch.nn.L1Loss(reduction='none')
    # print(loss(eng.feature_mat[0].unsqueeze(0), eng.feature_mat))


