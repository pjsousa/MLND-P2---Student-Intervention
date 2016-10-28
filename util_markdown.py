from __future__ import division
import math as math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd




def arr_to_markdown(arr):
    _grid_header = "\
| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |\n\
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |\n"

    step1_concat = [" | ".join([`r[1]`, `r[2]`, `r[5]`, `r[4]`, `r[6]`]) for r in arr]
    step2_pad = " |\n| ".join(r for r in step1_concat)

    _grid_content = "{}{}{}".format("| ", step2_pad, " |")
    
    return _grid_header + _grid_content

def dframe_to_markdown(df):
    _r = ""
    for itr_clf in df["classifier"].unique():
        _slice = df[df["classifier"] == itr_clf]
        _gridmarkdown = arr_to_markdown(_slice.as_matrix())
        _mkd_str = "\n\n\n** Classifer - {}**  \n\n{}".format(itr_clf, _gridmarkdown)
        _r = "{}{}".format(_r, _mkd_str)
        
    return _r
    

def render_scores (df):
    _side = int(math.ceil(math.sqrt(len(df["classifier"].unique()))))
    
    fig, axes = plt.subplots(nrows=_side, ncols=_side, figsize=[12,6*_side], squeeze=True)

    for idx, itr in enumerate(df["classifier"].unique()):
        _row = int(math.floor( idx / _side))
        _col = idx % _side
        
        g = df[df["classifier"] == itr]
        g.plot(x='test_size',y=['f1_train','f1_test'], ax=axes[_row, _col], yticks=np.arange(0,1.1,0.1))
        axes[_row, _col].set_title(itr)

        
def render_times (df):
    g = sns.factorplot(x="test_size", y="train_time_t", hue="classifier", data=df )
    (g.set_axis_labels("", "Train Time"))
    g = sns.factorplot(x="test_size", y="pred_time_tst", hue="classifier", data=df )
    (g.set_axis_labels("", "Test Predicting Time"))

