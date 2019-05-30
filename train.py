import os
from sklearn.model_selection import train_test_split
from models import ConvNet, Recurrent2DConvNet
import numpy as np
from data_generator import DataGenerator
from tensorboard_callbacks import TrainValTensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle


def build_train_test_split(args, use_random_val=True):
    '''
    creates lists of paths and labels, which are spaced by total duration.

    use_random_val: bool
        use the random order data as test split
    '''

    classes = sorted(['anxiety', 'baseline', 'concentration', 'digestion', 'disgust', 'frustration'])
    n_classes = len(classes)
    int2cls = dict(zip(range(len(classes)), classes))
    cls2int = dict(zip(classes, range(len(classes))))

    path = os.path.join(args.data_root, args.feats)
    paths = []
    val_paths = []
    labels = []
    val_labels = []

    for sub_dir in os.listdir(path):
        for class_dir in os.listdir(os.path.join(path, sub_dir)):
            cls_path = os.path.join(path, sub_dir, class_dir)
            files = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0]))
            if len(files) < args.total_dur:
                continue
            mod = len(files)%args.total_dur
            orig_len = len(files)
            for i in range(0, orig_len-mod-args.total_dur, args.total_dur):
                if use_random_val & ('r' in sub_dir):
                    val_paths.append(os.path.join(cls_path, files[i]))
                    val_labels.append(cls2int[class_dir])
                else:
                    paths.append(os.path.join(cls_path, files[i]))
                    labels.append(cls2int[class_dir])

    if val_paths != []:
        X_train, y_train = shuffle(paths, labels, random_state=0)
        X_test, y_test = shuffle(val_paths, val_labels, random_state=0)

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            paths, labels, test_size=0.1, random_state=0)

    return X_train, X_test, y_train, y_test


def train(args):

    classes = sorted(['anxiety', 'baseline', 'concentration', 'digestion', 'disgust', 'frustration'])
    n_classes = len(classes)

    X_train, X_test, y_train, y_test = build_train_test_split(args, use_random_val=False)

    # data is of shape (100, 128) or (100, 13) for 1 second of data
    # (time, feats)
    sample = np.load(X_train[0])

    print('\nTraining {} using {} features'.format(args.model, args.feats))

    if args.model is 'CNN':
        print('Using a total duration of {} seconds per sample\n'.format(args.total_dur))
        input_shape = (sample.shape[0]*args.total_dur, sample.shape[1], 1)
        model = ConvNet(input_shape=input_shape)

    elif args.model is 'RCNN':
        print('Using a total duration of {} seconds per sample'.format(args.total_dur),\
              'with time features every {} seconds'.format(args.delta_time), '\n')
        feat_dim = int(args.delta_time*100)
        time_dim = int(args.total_dur*sample.shape[0]/feat_dim)
        # (10, 30, 128, 1)
        input_shape = (time_dim, feat_dim, sample.shape[1], 1)
        model = Recurrent2DConvNet(input_shape=input_shape)

    model.summary()
    print('Input shape: {}'.format(input_shape))

    tg = DataGenerator(paths=X_train, targets=y_train, mode=args.model,
                       td=args.total_dur, dt=args.delta_time,
                       n_classes=n_classes, input_shape=input_shape)

    vg = DataGenerator(paths=X_test, targets=y_test, mode=args.model,
                       td=args.total_dur, dt=args.delta_time,
                       n_classes=n_classes, input_shape=input_shape)

    checkpoint = ModelCheckpoint(os.path.join('models', args.model+'.model'), monitor='val_acc', verbose=1, mode='max',
                                 save_best_only=True, save_weights_only=False, period=1)

    train_val = TrainValTensorBoard(write_graph=True)

    class_weight = compute_class_weight('balanced',
                                        np.unique(y_train),
                                        y_train)

    model.fit_generator(generator=tg, validation_data=vg,
                        steps_per_epoch=len(tg),
                        validation_steps=len(vg),
                        epochs=args.epochs, verbose=1,
                        class_weight=class_weight,
                        callbacks=[train_val, checkpoint])
