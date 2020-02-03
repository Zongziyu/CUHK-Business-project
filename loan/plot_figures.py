import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

def plotBar(dataset):
    matplotlib.style.use('ggplot')
    plt.figure(figsize=(15, 10))
    sns.barplot(x='purpose', y='profit_or_loss', data=dataset, hue='loan_status', palette=sns.color_palette('RdYlBu', 7))
    plt.legend(bbox_to_anchor=(0.8, 0.03), loc=3, borderaxespad=0, markerscale=3, fontsize='large')
    plt.title('Different Purpose Lead to Profitable or Loss', size=18)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.17, top=0.91)

    plt.xticks(rotation=-40)
    plt.savefig('./imgs/bar.png')
    plt.show()

def plotHeatmap(dataset):
    matplotlib.style.use('ggplot')
    plt.figure(figsize=(12,12))
    plt.title('Correlation of Features', y=1.05, size=20)
    sns.heatmap(dataset.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True, fmt='.2f')
    plt.xticks(rotation=-90)  # 将字体进行旋转
    plt.subplots_adjust(right=1.0,left=0.17)
    plt.savefig('./imgs/heatmap.png')
    plt.show()

def plotScatter(data,plot_col):
    matplotlib.style.use('ggplot')
    g = sns.PairGrid(data,x_vars=plot_col[:3],y_vars='profit_or_loss',height=4,hue='loan_status')
    g.map(sns.scatterplot,palette='pastel')
    plt.subplots_adjust(left=0.10,right=0.8,top=0.85)
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0,markerscale=1.5,fontsize='large')
    plt.savefig('./imgs/scatter1.png')

    g2 = sns.PairGrid(data,x_vars=plot_col[3:6],y_vars='profit_or_loss',height=4,hue='loan_status')
    g2.map(sns.scatterplot,palette='pastel')
    plt.subplots_adjust(left=0.10, right=0.8, top=0.85)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, markerscale=1.5, fontsize='large')
    plt.savefig('./imgs/scatter2.png')

    g3 = sns.PairGrid(data,x_vars=plot_col[6:9],y_vars='profit_or_loss',height=4,hue='loan_status')
    g3.map(sns.scatterplot,palette='pastel')
    plt.subplots_adjust(left=0.10, right=0.8, top=0.85)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, markerscale=1.5, fontsize='large')
    plt.savefig('./imgs/scatter3.png')
    plt.show()

def plotROC(fpr, tpr, roc_auc, model_pre):
    matplotlib.style.use('ggplot')
    model_name = ['LogisticRegression','XGBRFClassifier','MLPClassifier']
    for i in range(len(model_pre)):
        plt.plot(fpr[i],tpr[i],label='%s (auc = %0.2f)' % (model_name[i],roc_auc[i]),linewidth=2)
    plt.legend(bbox_to_anchor=(0.5,0.03), loc=3, borderaxespad=0,markerscale=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.subplots_adjust(left=0.10, right=0.95, top=0.9)
    plt.savefig('./imgs/ROC.png')
    plt.show()
    pass

def plotPredict(z_pred_LR, z_pred_XGBF, z_pred_MLP, z_pred_EEC):
    # plot_size = np.array(predict_list).shape[0]
    # fig, ax = plt.subplots(nrows=1,ncols=plot_size)
    # x = np.arange(len(predict))
    # print(x)
    # print(ax)
    # for i, j in zip(predict_list,range(plot_size)):
    #     print(i,j)
    #     ax[j]=sns.barplot(x=x,y=i,palette='pastel')
    #     plt.xlabel(model_name[j])
    x = np.arange(0,2*len(z_pred_EEC),2)
    print(x)
    x_label=[]
    for i in range(len(z_pred_EEC)):
        x_label.append('G'+str(i))
    print(x_label)
    fig, ax = plt.subplots()
    model_name = ['LogisticRegression', 'XGBRFClassifier', 'MLPClassifier','EasyEnsemble']
    width = 0.2
    plt.bar(x-width, z_pred_LR, width, label=model_name[0],color='tomato')
    plt.bar(x-width/2, z_pred_XGBF, width, label=model_name[1],color='darkorange')
    plt.bar(x+width/2, z_pred_MLP, width, label=model_name[2],color='turquoise')
    plt.bar(x+width, z_pred_EEC, width, label=model_name[3],color='deepskyblue')

    plt.title('Predict Classification Bar plot')
    ax.set_xticklabels(x_label), plt.ylabel('predict classification')
    plt.legend()
    plt.show()
    plt.savefig('./imgs/predict.png')
    pass