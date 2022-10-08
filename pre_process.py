import numpy as np
import csv
import random
import gzip
import torch
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader

nummotif = 16  # number of motifs to discover
bases = 'ACGT'  # DNA bases
basesRNA = 'ACGU'  # RNA bases
batch_size = 64  # fixed batch size -> see notes to problem about it
dictReverse = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}  # dictionary to implement reverse-complement mode
reverse_mode = False


def seqtopad(sequence, motlen, kind='DNA'):
    rows = len(sequence) + 2 * motlen - 2
    S = np.empty([rows, 4])
    base = bases if kind == 'DNA' else basesRNA
    for i in range(rows):
        for j in range(4):
            if i - motlen + 1 < len(sequence) and sequence[i - motlen + 1] == 'N' or i < motlen - 1 or i > len(
                    sequence) + motlen - 2:
                S[i, j] = np.float32(0.25)
            elif sequence[i - motlen + 1] == base[j]:
                S[i, j] = np.float32(1)
            else:
                S[i, j] = np.float32(0)
    return np.transpose(S)


def dinucshuffle(sequence):
    b = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d


def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complseq = [complement[base] for base in seq]
    return complseq


def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


class Chip:
    def __init__(self, filename, motiflen=24, reverse_complemet_mode=reverse_mode):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode = reverse_complemet_mode

    def openFile(self):
        train_dataset = []
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data, delimiter='\t')
            if not self.reverse_complemet_mode:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
            else:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append([seqtopad(reverse_complement(row[2]), self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
                    train_dataset.append([seqtopad(dinucshuffle(reverse_complement(row[2])), self.motiflen), [0]])
        # random.shuffle(train_dataset)
        train_dataset_pad = train_dataset

        size = int(len(train_dataset_pad) / 3)
        firstvalid = train_dataset_pad[:size]
        secondvalid = train_dataset_pad[size:size + size]
        thirdvalid = train_dataset_pad[size + size:]
        firsttrain = secondvalid + thirdvalid
        secondtrain = firstvalid + thirdvalid
        thirdtrain = firstvalid + secondvalid
        return firsttrain, firstvalid, secondtrain, secondvalid, thirdtrain, thirdvalid, train_dataset_pad


class chipseq_dataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self, xy=None):
        self.x_data = np.asarray([el[0] for el in xy], dtype=np.float32)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    chipseq = Chip('data/encode/ELK1_GM12878_ELK1_(1277-1)_Stanford_AC.seq.gz')

    train1, valid1, train2, valid2, train3, valid3, alldataset = chipseq.openFile()