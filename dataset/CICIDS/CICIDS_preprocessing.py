import numpy as np

from sklearn.model_selection import train_test_split

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

from utils import set_seed


class Preprocessor:
    def __init__(self, dataset_path, save_path, label_col_name, norm_method):
        self.df_path = None
        self.DataFrame = None
        self.test_df = None
        self.train_df = None
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.label_col_name = label_col_name
        self.norm_method = norm_method

    def __getitem__(self):
        return self.train_df, self.test_df

    def preprocess(self):
        self._read_data()
        self._rename()
        self._replace_labels()
        self._sampling()
        self._handle_missing()
        self._categorize_columns()
        self._scaling()
        self._reorder_columns()
        self._sort()
        self._shuffle()
        self._save()

    def _read_data(self):
        df = {}
        total = 0
        for idx, file in enumerate(os.listdir(self.dataset_path)):
            df[idx] = pd.read_csv(Path(self.dataset_path).joinpath(file))
            if idx == 0:
                self.DataFrame = pd.concat([self.DataFrame, df[idx]], axis=0)
            else:
                if df[idx].columns.all() == self.DataFrame.columns.all():
                    self.DataFrame = pd.concat([self.DataFrame, df[idx]], axis=0)
            print(file, f'= {len(df[idx])} samples')

            df_size = len(df[idx])
            total += df_size

        print(F'TOTAL NUMBER OF SAMPLES IN AGGREGATED DATAFRAME IS: {total}')

    def _rename(self):
        self.DataFrame.rename(
            columns={' Flow Packets/s': 'Flow_Packets',
                     'Flow Bytes/s': 'Flow_Bytes',
                     ' Label': str(self.label_col_name)},
            inplace=True)

    def _sampling(self):
        self.DataFrame, _ = train_test_split(self.DataFrame,
                                             test_size=1 - (1000000 / len(self.DataFrame)),
                                             stratify=self.DataFrame[self.label_col_name])

        X = self.DataFrame.iloc[:, :-1]
        y = self.DataFrame.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, stratify=y)
        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)

    def _categorize_columns(self):
        self.listBinary = [
            'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 'FIN Flag Count',
            ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
            ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', 'Fwd Avg Bytes/Bulk',
            ' Fwd Avg Packets/Bulk',
            ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
        self.listNominal = []

        columns = self.train_df.columns
        self.listNumerical = set(columns) - set(self.listNominal) - set(self.listBinary)
        self.listNumerical.remove(self.label_col_name)

    def _handle_missing(self):
        self.train_df['Flow_Bytes'].fillna(0, inplace=True)
        self.test_df['Flow_Bytes'].fillna(0, inplace=True)
        for col in self.train_df.columns[self.train_df.isin(['Infinity', np.inf, -np.inf]).any()]:
            max_val = self.train_df[col].replace([np.inf, -np.inf], np.nan).max()
            min_val = self.train_df[col].replace([np.inf, -np.inf], np.nan).min()

            self.train_df[col] = self.train_df[col].replace(np.inf, max_val)
            self.train_df[col] = self.train_df[col].replace(-np.inf, min_val)

            self.test_df[col] = self.test_df[col].replace(np.inf, max_val)
            self.test_df[col] = self.test_df[col].replace(-np.inf, min_val)

        train_columns_with_inf = self.train_df.columns[self.train_df.isin(['Infinity', np.inf, -np.inf, np.nan]).any()]
        test_columns_with_inf = self.test_df.columns[self.test_df.isin(['Infinity', np.inf, -np.inf, np.nan]).any()]
        if len(train_columns_with_inf) > 0:
            raise ValueError(F"INFINITE VALUES DETECTED IN {train_columns_with_inf} TRAIN COLUMNS.")
        if len(test_columns_with_inf) > 0:
            raise ValueError(F"INFINITE VALUES DETECTED IN {test_columns_with_inf} TEST COLUMNS.")

    def _scaling(self):
        train, test = self.train_df, self.test_df
        listContent = list(self.listNumerical)
        if self.norm_method == 'normalization':
            scaler = MinMaxScaler()
        elif self.norm_method == 'standardization':
            scaler = StandardScaler()
        scaler.fit(train[listContent].values)
        train[listContent] = scaler.transform(train[listContent])
        test[listContent] = scaler.transform(test[listContent])
        self.train_df, self.test_df = train, test

    def _replace_labels(self):
        print('LABELS OF DATASET BEFORE REPLACING LABELS\n', self.DataFrame[self.label_col_name].value_counts())
        self.DataFrame[self.label_col_name] = self.DataFrame[self.label_col_name].apply(
            lambda x: 1 if x != 'BENIGN' else 0)
        print('LABELS OF DATASET AFTER REPLACING LABELS\n', self.DataFrame[self.label_col_name].value_counts())

    def _reorder_columns(self):
        self.train_df = self.train_df[[col for col in self.train_df.columns
                                       if col != self.label_col_name] + [self.label_col_name]]
        self.test_df = self.test_df[[col for col in self.test_df.columns
                                     if col != self.label_col_name] + [self.label_col_name]]

    def _sort(self):
        train_cols = self.train_df.columns
        test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
        for col in test_sortedBasedOnTrain:
            test_sortedBasedOnTrain[col] = self.test_df[col]

        self.test_df = test_sortedBasedOnTrain

    def _shuffle(self):
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)

    def _save(self):
        self.train_df.to_csv(os.path.join(self.save_path, 'train_binary.csv'), index=False)
        self.test_df.to_csv(os.path.join(self.save_path, 'test_binary.csv'), index=False)


if __name__ == '__main__':
    set_seed(0)
    base_path = Path(__file__).resolve().parent.joinpath('file')
    data_path = base_path.joinpath('original')
    save_path = base_path.joinpath('preprocessed')

    preprocessor = Preprocessor(dataset_path=data_path,
                                save_path=save_path,
                                label_col_name='classification',
                                norm_method='standardization')
    preprocessor.preprocess()
    train_preprocessed, test_preprocessed = preprocessor.__getitem__()
    print(train_preprocessed.head())
    print(test_preprocessed.head())
