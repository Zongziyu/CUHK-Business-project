import numpy as np
import pandas as pd
import matplotlib
import os

from data_pre import data_load_proc
from learn import *
from plot_figures import *

def init_set():
    np.set_printoptions(suppress=True, threshold=1e4, linewidth=400)
    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    matplotlib.style.use('ggplot')

str_columns = ['home_ownership','loan_status','purpose','addr_state','earliest_cr_line']
classification_columns = ['term','id','delinq_2yrs','emp_length']
del_col = ['profit_or_loss','total_rec_prncp','total_rec_int','loan_amnt','total_pymnt']


if __name__ == '__main__':
    filepath = 'loan.csv'
    init_set()
    data=data_load_proc(filepath)
    # need_to_del_col = classification_columns + del_col + str_columns
    # plot_columns = data.columns - classification_columns - del_col - str_columns
    # plot_columns = data.columns.remove[classification_columns].remove[del_col].remove[str_columns]
    plot_columns = data.columns.difference(classification_columns + del_col + str_columns)
    print(plot_columns)
    # plotBar(data)
    # plotHeatmap(data)
    # plotScatter(data,plot_columns)
    index_ori, unindex_ori, index, unindex, x, y, z, x_train, x_test, y_train, y_test=preprocess(data)

    xx_train, xx_test, yy_train, yy_test, xx, yy, zz = preprocessing_cluster(data)

    # export_graphviz(tree, out_file="tree.dot", class_names=['good', 'bad'], feature_names=useful_columns[:-1],
    #                 impurity=False, filled=True)
    # # 展示可视化图
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('tree.png')

    y_pred_LR, y_pred_XGBF, y_pred_MLP, z_pred_LR, z_pred_XGBF, z_pred_MLP = learn_model(x_train,x_test,y_train,y_test,x,y,z)

    # print("--------------------------------------------------------------------")
    # _, _, _, _, _, _ = learn_model(xx_train, xx_test, yy_train, yy_test, xx, yy, zz)
    #
    #
    # fpr, tpr, roc_auc, model_pre = ROC(y_pred_LR, y_pred_XGBF, y_pred_MLP, y_test)
    # x_train, x_test, y_train, y_test, z = preprocessing_im(data)

    # z_pred_EEC = learn_imbalance(x_train,x_test,y_train,y_test,z)
    # plotPredict(z_pred_LR, z_pred_XGBF, z_pred_MLP, z_pred_EEC)
    # plotROC(fpr, tpr, roc_auc, model_pre)
