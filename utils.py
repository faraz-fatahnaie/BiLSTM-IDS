from sklearn import metrics
import numpy as np
from numpy import ndarray
from pathlib import Path
import os
import pandas as pd
from pandas import DataFrame
import tensorflow as tf
from keras.optimizers import Adam, SGD


def parse_data(df, dataset_name: str, classification_mode: str, mode: str = 'np'):
    classes = []
    if classification_mode == 'binary':
        classes = df.columns[-1:]
    elif classification_mode == 'multi':
        if dataset_name in ['NSL_KDD', 'KDD_CUP99']:
            classes = df.columns[-5:]
        elif dataset_name == 'UNSW_NB15':
            classes = df.columns[-10:]
        elif dataset_name == 'CICIDS':
            classes = df.columns[-15:]

    assert classes is not None, 'Something Wrong!!\nno class columns could be extracted from dataframe'
    glob_cl = set(range(len(df.columns)))
    cl_idx = set([df.columns.get_loc(c) for c in list(classes)])
    target_feature_idx = list(glob_cl.difference(cl_idx))
    cl_idx = list(cl_idx)
    dt = df.iloc[:, target_feature_idx]
    lb = df.iloc[:, cl_idx]
    assert len(dt) == len(lb), 'Something Wrong!!\nnumber of data is not equal to labels'
    if mode == 'np':
        return dt.to_numpy(), lb.to_numpy()
    elif mode == 'df':
        return dt, lb


def save_dataframe(dataframe: DataFrame, save_path: Path, dataframe_type: str = 'train',
                   classification_mode: str = 'binary') -> None:
    file_name = dataframe_type
    if classification_mode == 'binary':
        file_name = file_name + '_binary'
    elif classification_mode == 'multi':
        file_name = file_name + '_multi'
    train_file = os.path.join(save_path, file_name + '.csv')
    dataframe.to_csv(train_file, index=False)
    print('Saved:', train_file)


def sort_columns(train_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_cols = train_df.columns
    test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
    for col in test_sortedBasedOnTrain:
        test_sortedBasedOnTrain[col] = test_df[col]

    return train_df, test_sortedBasedOnTrain


def shuffle_dataframe(dataframe: DataFrame):
    return dataframe.sample(frac=1).reset_index(drop=True)


def metrics_evaluate(true_label: ndarray, pred_label: ndarray) -> dict:
    confusion_matrix = metrics.confusion_matrix(true_label, pred_label)
    metric_param = {'accuracy': metrics.accuracy_score(true_label, pred_label),
                    'f1_score': (metrics.f1_score(true_label, pred_label)),
                    'recall': metrics.recall_score(true_label, pred_label, average=None),
                    'confusion_matrix': confusion_matrix.tolist()}
    false_positive = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    false_negative = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    true_positive = np.diag(confusion_matrix)
    true_negative = confusion_matrix.sum() - (false_positive + false_negative + true_positive)

    metric_param['false_alarm_rate'] = (false_positive / (false_positive + true_negative)).tolist()
    metric_param['detection_rate'] = (true_positive / (true_positive + false_negative)).tolist()

    return metric_param


def set_seed(seed):
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


class OptimizerFactory:
    def __init__(self,
                 opt: str = 'adam',
                 lr_schedule: bool = True,
                 len_dataset: int = 494021,
                 epochs: int = 50,
                 batch_size: int = 100,
                 init_lr: float = 0.1,
                 final_lr: float = 0.00001):
        self.opt = opt
        self.lr_schedule = lr_schedule
        self.len_dataset = len_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.final_lr = final_lr

    def lr_scheduler(self):
        pretraining_learning_rate_decay_factor = (self.final_lr / self.init_lr) ** (1 / self.epochs)
        pretraining_steps_per_epoch = int(self.len_dataset / self.batch_size)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=pretraining_steps_per_epoch,
            decay_rate=pretraining_learning_rate_decay_factor,
            staircase=True)
        return lr_scheduler

    def get_opt(self):
        if self.opt == 'adam':
            if self.lr_schedule:
                return Adam(self.lr_scheduler())
            else:
                return Adam(learning_rate=self.init_lr)
        elif self.opt == 'sgd':
            if self.lr_schedule:
                return SGD(self.lr_scheduler())
            else:
                return SGD(learning_rate=5, decay=0.5, momentum=.85, nesterov=True)
                # return SGD(learning_rate=.1, decay=0.001, momentum=.95, nesterov=True)
