import argparse
import csv
import gc
from itertools import product

import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from sklearn import metrics
import tensorflow as tf
import os

from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dropout

from utils import parse_data, OptimizerFactory, get_result
from keras.layers import Input, LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, MultiHeadAttention
from keras.models import Model
from keras.optimizers import Adam
from utils import set_seed

# Set GPU device and disable eager execution
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# physical_devices = tf.config.list_physical_devices('GPU')

config = dict()

XGlobal = list()
YGlobal = list()

XValGlobal = list()
YValGlobal = list()

XTestGlobal = list()
YTestGlobal = list()

SavedParameters = list()
SavedParametersAE = list()
Mode = str()
Name = str()
SAVE_PATH_ = str()
result_path = str()
CHECKPOINT_PATH_ = str()

tid = 0
best_loss = float('inf')
best_val_acc = 0
best_ae = None
best_params = dict()
load_previous_result = True
continue_loading = True

set_seed(seed=0)


def train_cf(params):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global tid
    global best_ae

    global best_val_acc
    global SavedParameters

    global result_path
    global load_previous_result
    global continue_loading

    if (result_path is not None) and continue_loading:
        result_table = pd.read_csv(result_path)

        tid += 1
        selected_row = result_table[result_table['tid'] == params['tid']]
        print(selected_row)
        loss_hp = selected_row['F1_val'].values[0]
        loss_hp = -loss_hp
        if tid == len(result_table):
            continue_loading = False

        if load_previous_result:
            best_val_acc = result_table['F1_val'].max()

            result_table = result_table.sort_values('F1_val', ascending=False)
            SavedParameters = result_table.to_dict(orient='records')
            with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)

            load_previous_result = False

    else:
        tid += 1
        tf.keras.backend.clear_session()

        query_input = Input(shape=(XGlobal.shape[1],), dtype='float64')

        x = Embedding(XGlobal.shape[0], input_length=XGlobal.shape[1], output_dim=params['embed_dim'])(query_input)
        x = Dropout(params['dropout1'])(x)
        x = MultiHeadAttention(num_heads=params['n_head'], key_dim=params['embed_dim'])(x, x, x)

        # forward_layer = LSTM(params['LSTM_unit'])
        # backward_layer = LSTM(params['LSTM_unit'], go_backwards=True)
        # x = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='sum')(x)
        x = Bidirectional(LSTM(params['LSTM_unit']))(x)

        x = Dense(params['dense_unit'], activation='relu')(x)
        x = Dropout(params['dropout2'])(x)
        x = Dense(2, activation='softmax')(x)

        model = Model(inputs=query_input, outputs=x)

        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_variables])

        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH_, f'best_model.h5'),
            monitor='val_loss',
            mode='auto',
            save_best_only=True)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            mode="auto",
            restore_best_weights=True
        )

        cf_start = time.time()
        history = model.fit(XGlobal, YGlobal,
                            validation_data=(XValGlobal, YValGlobal),
                            epochs=100,
                            batch_size=params['batch_size'],
                            callbacks=[model_checkpoint, early_stop],
                            verbose=2)
        cf_time = (time.time() - cf_start)

        Y_predicted = model.predict(XValGlobal, workers=4, verbose=2)

        y_val = np.argmax(YValGlobal, axis=1)
        Y_predicted = np.argmax(Y_predicted, axis=1)

        cf_val = metrics.confusion_matrix(y_val, Y_predicted)
        results_val = get_result(cf_val)

        test_start_time = time.time()
        pred = model.predict(XTestGlobal, workers=4, verbose=2)
        test_elapsed_time = time.time() - test_start_time

        pred = np.argmax(pred, axis=1)
        y_eval = np.argmax(YTestGlobal, axis=1)

        cf_test = metrics.confusion_matrix(y_eval, pred)
        results_test = get_result(cf_test)

        result = {
            "tid": tid,
            "n_params": trainable_params,
            "n_head": params["n_head"],
            "embed_dim": params['embed_dim'],
            "LSTM_unit": params['LSTM_unit'],
            "dense_unit": params['dense_unit'],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "dropout1": params["dropout1"],
            "dropout2": params["dropout2"],
            "cf_time": cf_time,
            "TP_val": cf_val[0][0],
            "FP_val": cf_val[0][1],
            "TN_val": cf_val[1][1],
            "FN_val": cf_val[1][0],
            "OA_val": results_val['OA'],
            "P_val": results_val['P'],
            "R_val": results_val['R'],
            "F1_val": results_val['F1'],
            "FAR_val": results_val['FAR'],
            "test_time": int(test_elapsed_time),
            "TP_test": cf_test[0][0],
            "FP_test": cf_test[0][1],
            "FN_test": cf_test[1][0],
            "TN_test": cf_test[1][1],
            "OA_test": results_test['OA'],
            "P_test": results_test['P'],
            "R_test": results_test['R'],
            "F1_test": results_test['F1'],
            "FAR_test": results_test['FAR'],
        }

        SavedParameters.append(result)

        if SavedParameters[-1]["F1_val"] > best_val_acc:
            print("new model saved:" + str(SavedParameters[-1]))
            model.save(os.path.join(SAVE_PATH_, Name.replace(".csv", "_model.h5")))
            del model
            best_val_acc = SavedParameters[-1]["F1_val"]

        SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)

        try:
            with open((os.path.join(SAVE_PATH_, 'best_result.csv')), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
                writer.writeheader()
                writer.writerows(SavedParameters)
        except IOError:
            print("I/O error")

        loss_hp = -result["F1_val"]
        gc.collect()
    return {'loss': loss_hp, 'status': STATUS_OK}


def hyperparameter_tuning(dataset_name):
    global YGlobal
    global YValGlobal
    global YTestGlobal
    global XGlobal
    global XValGlobal
    global XTestGlobal

    global best_ae
    global best_params

    global SAVE_PATH_
    global CHECKPOINT_PATH_

    BASE_DIR = Path(__file__).resolve().parent
    BASE_DIR.joinpath('session').mkdir(exist_ok=True)

    i = 1
    flag = True

    while flag:

        TEMP_FILENAME = f"{dataset_name}-binary-MHA-{i}"
        TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

        if os.path.isdir(TEMP_PATH):
            i += 1
        else:
            flag = False

            os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
            SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

            os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/")

            print(f'MODEL SESSION: {SAVE_PATH_}')

    # Load and preprocess the training and testing data
    train = pd.read_csv(
        os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'train_binary.csv'))
    test = pd.read_csv(os.path.join(BASE_DIR, 'dataset', f'{dataset_name}', 'file', 'preprocessed', 'test_binary.csv'))

    XGlobal, YGlobal = parse_data(train, dataset_name=dataset_name, mode='np',
                                  classification_mode='binary')
    XTestGlobal, YTestGlobal = parse_data(test, dataset_name=dataset_name, mode='np',
                                          classification_mode='binary')

    YGlobal = tf.keras.utils.to_categorical(YGlobal, num_classes=2)
    YTestGlobal = tf.keras.utils.to_categorical(YTestGlobal, num_classes=2)

    XGlobal, XValGlobal, YGlobal, YValGlobal = train_test_split(XGlobal,
                                                                YGlobal,
                                                                test_size=0.2,
                                                                stratify=YGlobal,
                                                                random_state=0
                                                                )

    print('train set:', XGlobal.shape, YGlobal.shape)
    print('validation set:', XValGlobal.shape, YValGlobal.shape)
    print('test set:', XTestGlobal.shape, YTestGlobal.shape)

    cf_hyperparameters = {
        "n_head": hp.choice("n_head", [3, 4, 5]),
        "embed_dim": hp.choice("embed_dim", [32, 64, 128]),
        "LSTM_unit": hp.choice("LSTM_unit", [32, 64, 128]),
        "dense_unit": hp.choice("dense_unit", [64, 128, 256]),
        'dropout1': hp.uniform("dropout1", 0, 1),
        'dropout2': hp.uniform("dropout2", 0, 1),
        "batch_size": hp.choice("batch_size", [128, 256, 512]),
        "learning_rate": hp.uniform("learning_rate", 0.00001, 0.01)
    }
    trials = Trials()
    # spark_trials = SparkTrials()
    fmin(train_cf, cf_hyperparameters,
         trials=trials,
         algo=tpe.suggest,
         max_evals=30,
         rstate=np.random.default_rng(0))
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset', type=str, default='UNSW_NB15', required=True,
                        help='dataset name choose from: "UNSW", "KDD", "CICIDS"')
    parser.add_argument('--result', type=str, required=False,
                        help='path of hyper-parameter training result table .csv file')

    args = parser.parse_args()

    if args.result is not None:
        result_path = args.result
    else:
        result_path = None

    hyperparameter_tuning(args.dataset)
