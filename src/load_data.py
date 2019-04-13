#!/usr/bin/python
# -*- Coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from nklearn.encoder.onehot import GetOnehot

class LoadData:

    def __init__(self, fin, ftest, finfo):
        self.fin = fin
        self.ftest = ftest
        self.finfo = finfo

    def get_col_inf(self, finfo):
        return np.loadtxt(finfo, delimiter=",", encoding="utf-8", dtype=str, skiprows=1)[:23]

    def read(self, fin, ftest):
        return pd.read_csv(fin, index_col=0, dtype=str), pd.read_csv(ftest, index_col=0, dtype=str)

    def apply_format(self, col_inf, df):
        for col, tp in zip(col_inf.T[0], col_inf.T[1]):
            df[col] = df[col].astype(tp)
        return df

    def load(self):
        col_inf = self.get_col_inf(self.finfo)
        train, test = self.read(self.fin, self.ftest)
        return self.apply_format(col_inf, train), self.apply_format(col_inf, test)

    def get_features_v1(self):
        col_inf = self.get_col_inf(self.finfo).T
        cols_str = col_inf[0][col_inf[1] == "str"]
        cols_flt = col_inf[0][col_inf[1] == "float"]
        train, test = self.load()
        train_y = train.y
        train_X = train.drop("y",1)

        gh = GetOnehot()
        gh.fit(train_X[cols_str])
        train_oh, test_oh = gh.get_onehot(train_X), gh.get_onehot(test)

        scaler = StandardScaler()
        scaler.fit(train_X[cols_flt])
        train_std, test_std = scaler.transform(train_X[cols_flt]), scaler.transform(test[cols_flt])
        colname_flt = ["%s_std"%c for c in cols_flt]
        train_std, test_std = pd.DataFrame(train_std, columns=colname_flt), pd.DataFrame(test_std, columns=colname_flt)
        return pd.concat([train_oh, train_std], axis=1), train_y, pd.concat([test_oh, test_std], axis=1)

# 
# entry point
# 
if __name__ == "__main__":
    docs = "../docs/"
    info = docs + "info/"
    finfo = info + "ddl.csv"

    data = "../data/"
    data_in = data + "in/"
    fin = data_in + "train_data.csv"

    loader = LoadData(fin, finfo)
    loader.load()