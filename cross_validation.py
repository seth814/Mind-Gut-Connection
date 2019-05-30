import os
from sklearn.model_selection import KFold
import numpy as np
from models import ConvNet, Recurrent2DConvNet
from data_generator import DataGenerator
from sklearn.utils.class_weight import compute_class_weight
import pickle


def path_label_helper(args, sub_dirs):

    classes = sorted(['anxiety', 'baseline', 'concentration', 'digestion', 'disgust', 'frustration'])
    n_classes = len(classes)
    int2cls = dict(zip(range(len(classes)), classes))
    cls2int = dict(zip(classes, range(len(classes))))

    path = os.path.join(args.data_root, args.feats)
    paths = []
    labels = []

    for sub_dir in sub_dirs:
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

    return paths, labels


def train_cv(args):

    classes = sorted(['anxiety', 'baseline', 'concentration', 'digestion', 'disgust', 'frustration'])
    n_classes = len(classes)
    path = os.path.join(args.data_root, args.feats)

    cv_scores = {}
    cv_scores['train'] = []
    cv_scores['test'] = []
    sub_dirs = np.array(os.listdir(path))
    n_splits = 10
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (train, test) in enumerate(kfold.split(sub_dirs)):
        print('Fold {} of {}'.format(i+1, n_splits))
        X_train, y_train = path_label_helper(args, sub_dirs[train])
        X_test, y_test = path_label_helper(args, sub_dirs[test])

        sample = np.load(os.path.join(X_train[0]))

        if args.model is 'CNN':
            input_shape = (sample.shape[0]*args.total_dur, sample.shape[1], 1)
            model = ConvNet(input_shape=input_shape)

        elif args.model is 'RCNN':
            feat_dim = int(args.delta_time*100)
            time_dim = int(args.total_dur*sample.shape[0]/feat_dim)
            # (10, 30, 128, 1)
            input_shape = (time_dim, feat_dim, sample.shape[1], 1)
            model = Recurrent2DConvNet(input_shape=input_shape)

        tg = DataGenerator(paths=X_train, targets=y_train, mode=args.model,
                           td=args.total_dur, dt=args.delta_time,
                           n_classes=n_classes, input_shape=input_shape)

        vg = DataGenerator(paths=X_test, targets=y_test, mode=args.model,
                           td=args.total_dur, dt=args.delta_time,
                           n_classes=n_classes, input_shape=input_shape)

        class_weight = compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)

        model.fit_generator(generator=tg,
                            epochs=args.epochs, verbose=1,
                            class_weight=class_weight)

        train_scores = model.evaluate(tg)
        test_scores = model.evaluate(vg)
        cv_scores['train'].append(train_scores[1])
        cv_scores['test'].append(test_scores[1])

    print(cv_scores)

    with open(os.path.join('results', 'cv_scores.pkl'), 'wb') as handle:
        pickle.dump(cv_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
