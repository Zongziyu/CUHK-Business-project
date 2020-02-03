from data_pre import *
from learn import *

def init_set():
    np.set_printoptions(suppress=True, threshold=1e4, linewidth=400)
    # pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # matplotlib.style.use('ggplot')

str_columns = ['home_ownership','loan_status','purpose','addr_state','earliest_cr_line']
classification_columns = ['term','id','delinq_2yrs','emp_length']
del_col = ['profit_or_loss','total_rec_prncp','total_rec_int','loan_amnt','total_pymnt']

if __name__ == '__main__':
    filepath = 'loan.csv'
    init_set()
    data=data_load_proc(filepath)
    plot_columns = data.columns - classification_columns - del_col - str_columns
    # plotBar(data)
    # plotHeatmap(data)
    # plotScatter(data,plot_columns)

    data = data[useful_columns]
    index = data[data['loan_status'] != 'Default']
    unindex = data[data['loan_status'] == 'Default']
    index['loan_status'] = index['loan_status'].map(loan_status_dict_2)
    positive = index[index['loan_status'] == 1]
    negative = index[index['loan_status'] == -1]
    x,y = np.array(positive[useful_columns[:-1]]), np.array(negative[useful_columns[:-1]])
    zz = np.array(unindex[useful_columns[:-1]])
    x, y, zz = scale(x), scale(y), scale(zz)

    print(x)
    #
    # if type == 1:
    cluster_p = KMeans(n_clusters=800).fit(x)
    cluster_n = KMeans(n_clusters=400).fit(y)
    print(cluster_n.labels_)
    p_c, n_c = cluster_p.cluster_centers_, cluster_n.cluster_centers_
    print('p_c',p_c,p_c.shape)
    print('n_c',n_c,n_c.shape)
    xx = np.vstack((p_c,n_c))
    yy = np.hstack((np.ones(800),np.ones(400)*(-1)))
    print(xx.shape,yy.shape)
    xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=33, shuffle=True)

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(xx_train, yy_train)
    cross_10_acc = cross_val_score(tree, xx, yy, cv=10)
    cross_10_pre = cross_val_score(tree, xx, yy, cv=10, scoring='precision')
    # print('DecisionTreeClassifier (train): ', tree.score(x_train, y_train))
    # print('DecisionTreeClassifier (test): ', tree.score(x_test, y_test))
    z_pre_tree = tree.predict(zz)
    print('DecisionTreeClassifier', cross_10_acc, np.mean(cross_10_acc), np.std(cross_10_acc, ddof=1))
    print('DecisionTreeClassifier', cross_10_pre, np.mean(cross_10_pre), np.std(cross_10_pre, ddof=1))
    print('DecisionTreeClassifier', z_pre_tree)