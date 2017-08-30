import argparse
import sys
import os
from pprint import pprint

VGG_PATH = 'data/train/imagenet-vgg-verydeep-19.mat'
TRAIN_DIR = 'data/train/train2014'

def append_train_args():
    args.train = 1
    args.style_name = os.path.splitext(os.path.basename(args.target_style))[0]
    args.model_name = '{:s}_sw={:d}'.format(args.style_name, int(args.style_weight))
    args.model_dir = os.path.join('checkpoints', args.model_name)
    args.log_dir = os.path.join('log', args.model_name)

def append_test_args():
    args.train = 0
    args.style_name = args.model_name.split('_sw=')[0]
    args.model_dir = os.path.join('checkpoints', args.model_name)

def check_args():
    pprint(vars(args))
    if args.train:
        try:
            os.makedirs(args.model_dir)
        except OSError:
            print "Overriding model directory"

    else:
        try:
            os.makedirs('results')
        except OSError:
            print "Using existing results directory"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers()

# Training args
parser_train = subparsers.add_parser('train')
parser_train.add_argument('--content-weight', type=float, default=1.5e1,                  help="Content weight")
parser_train.add_argument('--style-weight',   type=float, default=1e2,                    help="Style weight")
# Optimization
parser_train.add_argument('--learning-rate',  type=float, default=1e-3,                   help="Learning rate")
parser_train.add_argument('--n-epochs',       type=int,   default=10,                     help="No. epochs")
parser_train.add_argument('--batch-size',     type=int,   default=20,                     help="Batch size")
# Data (remember to change the paths)
parser_train.add_argument('--vgg-path',       type=str,   default=VGG_PATH,               help="Path to pre-trained VGG weights")
parser_train.add_argument('--train-dir',      type=str,   default=TRAIN_DIR,              help="Path to training data")
parser_train.add_argument('--target-style',   type=str,   default='data/styles/line.jpg', help="Style file")
parser_train.add_argument('--validation-dir', type=str,   default='data/validation',      help="Path to validation images")
# Training checkpoints
parser_train.add_argument('--iter-visualize', type=int,   default=1000,                   help="Visualize per iter")
parser_train.add_argument('--iter-save',      type=int,   default=5000,                   help="Save per iter")
parser_train.set_defaults(append=append_train_args)

# Test args
parser_test = subparsers.add_parser('test')
parser_test.add_argument('--model-name',      type=str, required=True,                     help="Name of model used for eval")
parser_test.add_argument('--ckpt',            type=int,                                    help="Which checkpoint to use")
parser_test.add_argument('--test-dir',        type=str,                                    help="Test directory")
parser_test.add_argument('--test-file',       type=str,                                    help="Test file")
parser_test.set_defaults(append=append_test_args)

if 'ipykernel' in sys.argv[0]:
    args = parser.parse_args(['train'])
else:
    args = parser.parse_args()

args.append()
check_args()
