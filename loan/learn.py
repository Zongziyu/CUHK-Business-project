from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier,XGBRFClassifier
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,roc_curve,auc
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
from sklearn.model_selection import cross_val_score
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift
import numpy as np
from plot_figures import plotPredict

useful_columns = ['int_rate','installment','emp_length','annual_inc','dti','delinq_2yrs','open_acc','revol_bal','total_pymnt','profit_or_loss','loan_status']

loan_status_dict_1 = {'Fully Paid':1, 'Current':2, 'Charged Off':3, 'Late (31-120 days)':4, 'In Grace Period':5, 'Late (16-30 days)':6, 'Default':0}
loan_status_dict_2 = {'Fully Paid':1, 'Current':1, 'Charged Off':-1, 'Late (31-120 days)':-1, 'In Grace Period':-1, 'Late (16-30 days)':-1, 'Default':-1}


def preprocess(data):
    data=data.sort_values(by = 'loan_status', ascending= True)
    index_ori = data[data['loan_status'] != 'Default']
    # print(index_ori)
    index_ori = index_ori[:218].append(index_ori[7500:8000]).append(index_ori[8300:8800]).append(index_ori[9200:])

    # print(len(useful_columns))
    # print(index_ori.shape)
    print(index_ori['loan_status'].value_counts())
    unindex_ori = data[data['loan_status'] == 'Default']
    index = index_ori[useful_columns]
    unindex = unindex_ori[useful_columns]
    index['loan_status'] = index['loan_status'].map(loan_status_dict_2)
    x = np.array(index[useful_columns[:-1]])
    y = np.array(index['loan_status'])

    # print(y.shape)

    # divide the dataset into train_set and test_set
    # to predict the rows (loan_status is 'Default')
    z = np.array(unindex[useful_columns[:-1]])  # 未标签数据
    x, z = scale(x), scale(z)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
    return index_ori,unindex_ori,index,unindex,x,y,z,x_train,x_test,y_train,y_test

def preprocessing_cluster(data):
    data = data[useful_columns]
    index = data[data['loan_status'] != 'Default']
    unindex = data[data['loan_status'] == 'Default']
    index['loan_status'] = index['loan_status'].map(loan_status_dict_2)
    positive = index[index['loan_status'] == 1]
    negative = index[index['loan_status'] == -1]
    x,y = np.array(positive[useful_columns[:-1]]), np.array(negative[useful_columns[:-1]])
    zz = np.array(unindex[useful_columns[:-1]])
    x, y, zz = scale(x), scale(y), scale(zz)
    #
    # if type == 1:
    cluster_p = KMeans(n_clusters=800).fit(x)
    cluster_n = KMeans(n_clusters=400).fit(y)
    p_c, n_c = cluster_p.cluster_centers_, cluster_n.cluster_centers_
    xx = np.vstack((p_c,n_c))
    yy = np.hstack((np.ones(800),np.ones(400)*(-1)))

    xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=33, shuffle=True)
    return xx_train, xx_test, yy_train, yy_test, xx, yy, zz
    #     pass
    # elif type == 2:
    #     cluster_p = MeanShift(n_jobs=)
    #     pass
    # elif type == 3:
    #     pass

def preprocessing_im(data):
    index_ori = data[data['loan_status'] != 'Default']
    unindex_ori = data[data['loan_status'] == 'Default']
    index = index_ori[useful_columns]
    unindex = unindex_ori[useful_columns]
    index['loan_status'] = index['loan_status'].map(loan_status_dict_2)
    x = np.array(index[useful_columns[:-1]])
    y = np.array(index['loan_status'])
    # divide the dataset into train_set and test_set
    # to predict the rows (loan_status is 'Default')
    z = np.array(unindex[useful_columns[:-1]])  # 未标签数据
    x, z = scale(x), scale(z)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
    return x_train,x_test,y_train,y_test,z

def learn_imbalance(x_train,x_test,y_train,y_test,z):
    EEC = EasyEnsembleClassifier(random_state=42)
    EEC.fit(x_train, y_train)
    z_pred_EEC = EEC.predict(z)
    y_pred_EEC = EEC.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_EEC)
    auc_ = auc(fpr, tpr)
    print('EasyEnsembleClassifier',EEC.score(x_test,y_test))
    print('EasyEnsembleClassifier',z_pred_EEC)
    print('EasyEnsemble\'s auc',auc_)

    return z_pred_EEC


def learn_model(x_train,x_test,y_train,y_test,x,y,z):

    fpr, tpr, roc_auc = [], [], []
    model_name = ['LogisticRegression','XGBRFClassifier','MLPClassifier']

    tree = DecisionTreeClassifier(random_state=0,max_depth=3,min_samples_leaf=5)
    tree.fit(x_train, y_train)
    cross_10_acc = cross_val_score(tree, x, y, cv=10)
    cross_10_pre = cross_val_score(tree, x, y, cv=10, scoring='precision')
    # print('DecisionTreeClassifier (train): ', tree.score(x_train, y_train))
    # print('DecisionTreeClassifier (test): ', tree.score(x_test, y_test))
    z_pre_tree = tree.predict(z)
    print('DecisionTreeClassifier', cross_10_acc, np.mean(cross_10_acc), np.std(cross_10_acc, ddof=1))
    print('DecisionTreeClassifier', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    print('DecisionTreeClassifier', z_pre_tree)

    LR = LogisticRegression()
    LR.fit(x_train, y_train)
    z_pred_LR = LR.predict(z)
    y_pred_LR = LR.decision_function(x_test)
    # print(y_pred_LR)
    fpr_, tpr_, _ = roc_curve(y_test,y_pred_LR)
    fpr.append(fpr_), tpr.append(tpr_),roc_auc.append(auc(fpr_,tpr_))
    cross_10_acc = cross_val_score(LR,x,y,cv=10)
    print('LogisticRegression',cross_10_acc,np.mean(cross_10_acc),np.std(cross_10_acc,ddof=1))
    cross_10_pre = cross_val_score(LR, x, y, cv=10, scoring='precision')
    print('LogisticRegression', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    # print('LogisticRegression',LR.score(x_test, y_test))
    print('LogisticRegression',z_pred_LR)

    # XGB = XGBClassifier()
    # XGB.fit(x_train, y_train)
    # z_pred = XGB.predict(z)
    # y_pred_XGB = XGB.predict(x_test)
    # fpr.append(fpr_), tpr.append(tpr_), roc_auc.append(auc(fpr_, tpr_))
    # print('XGBClassifier',XGB.score(x_test,y_test))
    # print('XGBClassifier',z_pred)

    XGBF = XGBRFClassifier()
    XGBF.fit(x_train, y_train)
    z_pred_XGBF = XGBF.predict(z)
    y_pred_XGBF = XGBF.predict(x_test)
    fpr.append(fpr_), tpr.append(tpr_), roc_auc.append(auc(fpr_, tpr_))
    cross_10_acc = cross_val_score(XGBF, x, y, cv=10)
    print('XGBRFClassifier', cross_10_acc, np.mean(cross_10_acc),np.std(cross_10_acc,ddof=1))
    cross_10_pre = cross_val_score(XGBF, x, y, cv=10, scoring='precision')
    print('XGBRFClassifier', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    # print('XGBRFClassifier',XGBF.score(x_test,y_test))
    print('XGBRFClassifier',z_pred_XGBF)

    MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(11, 2), random_state=1)
    MLP.fit(x_train,y_train)
    z_pred_MLP = MLP.predict(z)
    y_pred_MLP = MLP.predict(x_test)
    cross_10_acc = cross_val_score(MLP, x, y, cv=10)
    print('MLPClassifier2', cross_10_acc, np.mean(cross_10_acc),np.std(cross_10_acc,ddof=1))
    cross_10_pre = cross_val_score(MLP, x, y, cv=10, scoring='precision')
    print('MLPClassifier2', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    # print('MLPClassifier2',MLP.score(x_test,y_test))
    print('MLPClassifier2',z_pred_MLP)

    MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(11), random_state=1)
    MLP.fit(x_train, y_train)
    z_pred_MLP = MLP.predict(z)
    y_pred_MLP = MLP.predict(x_test)
    cross_10_acc = cross_val_score(MLP, x, y, cv=10)
    print('MLPClassifier1', cross_10_acc, np.mean(cross_10_acc), np.std(cross_10_acc, ddof=1))
    cross_10_pre = cross_val_score(MLP, x, y, cv=10, scoring='precision')
    print('MLPClassifier1', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    # print('MLPClassifier', MLP.score(x_test, y_test))
    print('MLPClassifier1', z_pred_MLP)

    return y_pred_LR, y_pred_XGBF, y_pred_MLP, z_pred_LR, z_pred_XGBF, z_pred_MLP

def ROC(y_pred_LR, y_pred_XGBF, y_pred_MLP, y_test):
    model_pre = ['y_pred_LR', 'y_pred_XGBF', 'y_pred_MLP']
    fpr, tpr, roc_auc = [], [], []
    # print('y_test',y_test)
    for i in range(len(model_pre)):
        fpr_, tpr_, _ = roc_curve(y_test,eval(model_pre[i]))
        fpr.append(fpr_)
        tpr.append(tpr_)
        roc_auc.append(auc(fpr_,tpr_))
        # print('fpr_',fpr_)
        # print('tpr_',tpr_)
        # print('_',_)
    # print('fpr',fpr)
    # print('tpr',tpr)
    # print('roc_auc',roc_auc)
    return fpr, tpr, roc_auc, model_pre