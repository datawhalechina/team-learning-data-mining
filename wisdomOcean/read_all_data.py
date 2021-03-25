# -*- codeing = utf-8 -*-
# @Time : 2021/3/7 22:23
# @Author : Evan_wyl
# @File : read_all_data.py
import pandas as pd


def read_train_file(filename=None):
    # 替换数据存放的路径
    Path = "D:/code_sea/data/train/hy_round1_train_20200102/"
    return pd.read_csv(Path + filename,encoding="utf-8")

def read_test_file(filename=None):
    # 替换数据存放的路径
    Path = "D:/code_sea/data/test/hy_round1_testA_20200102/"
    return pd.read_csv(Path + filename,encoding="utf-8")

