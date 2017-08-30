from config import args
from model import Model
from train import train
from test import test
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    model = Model()

    if args.train:
        train(model)

    else:
        test(model)

if __name__ == '__main__':
    main()
