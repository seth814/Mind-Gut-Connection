import argparse
from train import train

parser = argparse.ArgumentParser(description='TensorFlow Mind Gut Connection Model')

#parser.add_argument('model_dir', help='output directory to save models & results')

parser.add_argument('--data_root', type=str, default='/home/seth/datasets/gut',\
                    help='directory containing stft and mfcc dir')

parser.add_argument('-f', '--feats', type=str, default='stft',
                    help='name of features to use for modeling (stft or mfcc)')

parser.add_argument('-m', '--model', type=str, default='RCNN',
                    help='string of model to run (CNN or RCNN)')

parser.add_argument('-td', '--total_dur', type=int, default=3,
                    help='total time window (multiples of 1 second)')

parser.add_argument('-dt', '--delta_time', type=float, default=1.0,
                    help='duration for each time dimension (seconds) (for RCNN)')

parser.add_argument('-t', '--is_train', type=int, default=1,\
                    help='use 1 to train model')

parser.add_argument('-e', '--epochs', type=int, default=120,\
                    help='number of training epochs')

args = parser.parse_args()


def main():
    '''Trains a model to classify audio from intestine into activity states'''
    if args.is_train == 1:
        train(args)


if __name__ == '__main__':
    main()
