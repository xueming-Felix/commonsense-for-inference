import argparse
import os
import random

import torch

from utils import *


def train():

    dataset = load_data(dataset='commonsense_qa')
    # data_tr = dataset.data['train']

    return dataset


def test():
    pass


def main(params):
    if params.train:
        model = train()
        # torch.save(model, params.model_file)
    else:
        test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Common sense question answering")
    parser.add_argument("--train", action='store_const', const=True, default=True)

    main(parser.parse_args())
