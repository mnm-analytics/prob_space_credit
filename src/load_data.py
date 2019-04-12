#!/usr/bin/python
# -*- Coding: utf-8 -*-

import pandas as pd
import numpy as np

class LoadData:

    def __init__(self, fin, finfo):
        self.fin = fin
        self.finfo = finfo

    def get_col_inf(self, finfo):
        return np.loadtxt(finfo, delimiter=",", encoding="utf-8", dtype=str, skiprows=1)[:23]

    def read(self, fin):
        return pd.read_csv(fin, index_col=0, dtype=str)

    def apply_format(self, col_inf, df):
        for col, tp in zip(col_inf.T[0], col_inf.T[1]):
            df[col] = df[col].astype(tp)
        if "y" not in set(df.columns):
            return df
        else:
            return df.drop("y",1), df["y"]

    def load(self):
        col_inf = self.get_col_inf(self.finfo)
        df = self.read(self.fin)
        return self.apply_format(col_inf, df)
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