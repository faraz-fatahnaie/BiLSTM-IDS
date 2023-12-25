import argparse
import csv

import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from keras.utils import to_categorical
import os
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from BILSTM import BiLstm

from utils import parse_data
from configs.setting import setting
from utils import OptimizerFactory, set_seed

set_seed(seed=0)


def train(dataset_name):
    i = 1
    flag = True
    SAVE_PATH_ = ''
    TRAINED_MODEL_PATH_ = ''
    CHECKPOINT_PATH_ = ''
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:

        config, config_file = setting()
        TEMP_FILENAME = f"{dataset_name}-{config['CLASSIFICATION_MODE']}-{config['MODEL_NAME']}-{i}"
        TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
            SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

            os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/")

            with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
                json.dump(config_file, f)

            print(f'MODEL SESSION: {SAVE_PATH_}')

    # Load and preprocess the training and testing data
    train = pd.read_csv(
        os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'train_binary.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'test_binary.csv'))

    X_train, Y_train = parse_data(train, dataset_name=dataset_name, mode=config['DATASET_TYPE'],
                                  classification_mode=config['CLASSIFICATION_MODE'])
    X_test, Y_test = parse_data(test, dataset_name=dataset_name, mode=config['DATASET_TYPE'],
                                classification_mode=config['CLASSIFICATION_MODE'])

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1])).astype('float32')
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1])).astype('float32')
    # Y_train = to_categorical(Y_train, num_classes=2)
    Y_test = to_categorical(Y_test, num_classes=2)

    print('train set:', X_train.shape, Y_train.shape)
    print('validation set:', X_test.shape, Y_test.shape)

    loss_fn = {
        'cce': 'categorical_crossentropy',
        'bce': 'binary_crossentropy'
    }

    opt_factory = OptimizerFactory(opt=config['OPTIMIZER'],
                                   lr_schedule=True,
                                   len_dataset=len(X_train),
                                   epochs=config['EPOCHS'],
                                   batch_size=config['BATCH_SIZE'],
                                   init_lr=config['LR'],
                                   final_lr=config['MIN_LR'])

    skf = StratifiedKFold(n_splits=config['K_FOLD'], shuffle=True, random_state=config['SEED'])

    partial_metrics = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for index, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):
        print("Training on fold " + str(index + 1) + "/10...")

        x_train, x_val = X_train[train_indices], X_train[val_indices]
        y_train, y_val = Y_train[train_indices], Y_train[val_indices]

        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        model = BiLstm(in_shape=(x_train.shape[1],),
                       out_shape=y_train.shape[1],
                       length_in=len(x_train)).build_graph()

        model.compile(optimizer=opt_factory.get_opt(), loss=loss_fn[config['LOSS']], metrics=['accuracy'])

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH_, f'model_{index + 1}.h5'),
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.0001,
            patience=config['PATIENCE'],
            mode="auto",
            restore_best_weights=True
        )

        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=config['EPOCHS'],
                            batch_size=config['BATCH_SIZE'],
                            callbacks=[model_checkpoint, early_stop],
                            verbose=2)

        pred = model.predict(X_test, workers=4, verbose=2)

        pred = np.argmax(pred, axis=1)
        label = np.argmax(Y_test, axis=1)

        cf = metrics.confusion_matrix(label, pred)
        accuracy_scores.append(metrics.accuracy_score(label, pred))
        precision_scores.append(metrics.precision_score(label, pred, average='binary'))
        recall_scores.append(metrics.recall_score(label, pred, average='binary'))
        f1_scores.append(metrics.f1_score(label, pred, average='binary'))

        metrics_df = {
            "id": index + 1,
            "TP_val": cf[0][0],
            "FP_val": cf[0][1],
            "TN_val": cf[1][1],
            "FN_val": cf[1][0],
            'Accuracy': accuracy_scores[index],
            'Precision': precision_scores[index],
            'Recall': recall_scores[index],
            'F1 Score': f1_scores[index],
        }
        partial_metrics.append(metrics_df)
        try:
            with open((os.path.join(SAVE_PATH_, 'k-fold_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=partial_metrics[0].keys())
                writer.writeheader()
                writer.writerows(partial_metrics)
        except IOError:
            print("I/O error")

    final_metrics = []
    avg_metrics = {
        "avg_accuracy": sum(accuracy_scores) / config['K_FOLD'],
        "avg_precision": sum(precision_scores) / config['K_FOLD'],
        "avg_recall": sum(recall_scores) / config['K_FOLD'],
        "avg_f1": sum(f1_scores) / config['K_FOLD'],

    }
    final_metrics.append(avg_metrics)
    try:
        with open((os.path.join(SAVE_PATH_, 'final_result.csv')), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_metrics[0].keys())
            writer.writeheader()
            writer.writerows(final_metrics)
    except IOError:
        print("I/O error")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW_NB15', required=True,
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')
    args = parser.parse_args()

    train(args.dataset)
