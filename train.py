import os
from sklearn.model_selection import train_test_split
from models import ConvNet, Recurrent2DConvNet
import numpy as np
from data_generator import DataGenerator
from tensorboard_callbacks import TrainValTensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


def train(args):

    classes = sorted(['anxiety', 'baseline', 'concentration', 'digestion', 'disgust', 'frustration'])
    n_classes = len(classes)
    int2cls = dict(zip(range(len(classes)), classes))
    cls2int = dict(zip(classes, range(len(classes))))

    path = os.path.join(args.data_root, args.feats)
    paths = []
    labels = []

    for sub_dir in os.listdir(path):
        for class_dir in os.listdir(os.path.join(path, sub_dir)):
            cls_path = os.path.join(path, sub_dir, class_dir)
            files = sorted(os.listdir(cls_path), key=lambda x: int(x.split('.')[0]))
            if len(files) < args.total_dur:
                continue
            mod = len(files)%args.total_dur
            orig_len = len(files)
            for i in range(0, orig_len-mod-args.total_dur, args.total_dur):
                paths.append(os.path.join(cls_path, files[i]))
                labels.append(cls2int[class_dir])

    # data is of shape (100, 128) or (100, 13)
    # (time, feats)
    sample = np.load(os.path.join(cls_path, files[i]))

    print('\nTraining {} using {} features'.format(args.model, args.feats))

    if args.model is 'CNN':
        print('Using a total duration of {} seconds per sample\n'.format(args.total_dur))
        input_shape = (sample.shape[0]*args.total_dur, sample.shape[1], 1)
        print(input_shape)
        model = ConvNet(input_shape=input_shape)

    elif args.model is 'RCNN':
        print('Using a total duration of {} seconds per sample'.format(args.total_dur),\
              'with time features every {} seconds'.format(args.delta_time), '\n')
        feat_dim = int(args.delta_time*100)
        time_dim = int(args.total_dur*sample.shape[0]/feat_dim)
        # (10, 30, 128, 1)
        input_shape = (time_dim, feat_dim, sample.shape[1], 1)
        model = Recurrent2DConvNet(input_shape=input_shape)

    print('Input shape: {}'.format(input_shape))

    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels, test_size=0.1, random_state=0)

    tg = DataGenerator(paths=X_train, targets=y_train, mode=args.model,
                       td=args.total_dur, dt=args.delta_time, epoch_frac=0.25,
                       n_classes=n_classes, input_shape=input_shape)

    vg = DataGenerator(paths=X_test, targets=y_test, mode=args.model,
                       td=args.total_dur, dt=args.delta_time,
                       n_classes=n_classes, input_shape=input_shape)

    checkpoint = ModelCheckpoint(os.path.join('models', args.model+'.h5'), monitor='val_acc', verbose=1, mode='max',
                                 save_best_only=True, save_weights_only=True, period=1)

    train_val = TrainValTensorBoard(write_graph=True)

    class_weight = compute_class_weight('balanced',
                                        np.unique(labels),
                                        labels)

    model.fit_generator(generator=tg, validation_data=vg,
                        epochs=args.epochs, verbose=1,
                        class_weight=class_weight,
                        callbacks=[train_val, checkpoint])
