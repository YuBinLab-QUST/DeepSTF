import numpy as np


def one_hot(seq):

    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    code = np.empty(shape=(len(seq), 4))
    for location, base in enumerate(seq, start=0):
        code[location] = base_map[base]

    return code