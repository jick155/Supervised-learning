# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:25:43 2018

@author: ST
"""

def train_history_graphic( history       # 資料集合
                         , history_key1  # 資料集合裡面的來源 1 (有 loss, acc, val_loss, val_acc 四種)
                         , history_key2  # 資料集合裡面的來源 2 (有 loss, acc, val_loss, val_acc 四種)
                         , y_label       # Y 軸標籤文字
                         ) :
    import matplotlib.pyplot as plt
    plt.plot( history.history[history_key1] )    # 資料來源 1    
    plt.plot( history.history[history_key2] )   # 資料來源 2    
    plt.title( 'train history' )    # 標題    
    plt.xlabel( 'epochs' )  # X 軸標籤文字    
    plt.ylabel( y_label )   # Y 軸標籤文字
    # 設定圖例
    # (參數 1 為圖例說明, 有幾個資料來源, 就對應幾個圖例說明)
    # (參數 2 為圖例位置, upper 為上面, lower 為下面, left 為左邊, right 為右邊)
    plt.legend( ['train', 'validate'] , loc = 'upper left')   
    plt.show()# 顯示畫布
    


def show_feature_label_prediction( features
                                 , labels
                                 , predictions
                                 , indexList  # 資料集合中, 要顯示的索引陣列
                                 , label_desc
                                 ) :
    import math
    import matplotlib.pyplot as plt
    import numpy as np  
    num = len(indexList)
    plt.gcf().set_size_inches( 2*5, (2+0.4)*math.ceil(num/5) )     # 設定畫布的寬（參數 1）與高（參數 2）
    loc = 0
    for i in indexList :        
        loc += 1    # 目前要在畫布上的哪個位置顯示 (從 1 開始)        
        subp = plt.subplot( math.ceil(num/5), 5, loc ) # 畫布區分為幾列（參數 1）, 幾欄（參數 2）, 目前在哪個位置（參數 3）        
        subp.imshow( np.squeeze(features[i],axis=(2,)), cmap='binary' ) # 畫布上顯示圖案, 其中 cmap=binary 為顯示黑白圖案
        if( len(predictions) > 0 ) :# 設定標題內容,有 AI 預測結果資料, 才在標題顯示預測結果
            title = 'ai = ' + label_desc[ predictions[i] ]
            title += (' (o)' if predictions[i]==labels[i] else ' (x)') # 預測正確顯示(o), 錯誤顯示(x)
            title += '\nlabel = ' + label_desc[ labels[i] ]        
        else : # 沒有 AI 預測結果資料, 則只在標題顯示真實數值
            title = 'label = ' + label_desc[ labels[i] ]       
        subp.set_title( title, fontsize=12 ) # 在畫布上顯示標題, 且字型大小為 12       
        subp.set_xticks( [] )   # X, Y 軸不顯示刻度
        subp.set_yticks( [] )    
    plt.show() # 顯示畫布

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix'):
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def anomalyScatter(features,autoencoder):  #anomaly散佈圖 
    import numpy as np
    import pandas as pd
    predictions=autoencoder.predict(features)    
    mse=np.mean(np.power(features-predictions,2),axis=1)
    if mse.ndim>2:
        mse=np.sum(mse,axis=1)
        mse=(np.squeeze(mse))  
    mseDataframe=pd.DataFrame({'mse':mse})        
    #mseDataframe['outlier']=mseDataframe[(mseDataframe['mse']>outlierMse)]
    mseDataframe['index']=mseDataframe.index
    mseDataframe.plot(kind='scatter',x='index',y='mse',c='mse',ylim=(min(mse),max(mse)))
    
def anomalyHist(features,autoencoder):  #anomaly直方圖  
    import numpy as np
    import pandas as pd
    predictions=autoencoder.predict(features)
    mse=np.mean(np.power(features-predictions,2),axis=1)
    if mse.ndim>2:
        mse=np.sum(mse,axis=1)
        mse=(np.squeeze(mse))
    mseDataframe=pd.DataFrame({'mse':mse})    
    mseDataframe['mse'].hist(bins=100,range=[min(mseDataframe['mse']),max(mseDataframe['mse'])])

def raxImgVsCoderImg(features,autoencoder,zoom,outlierMse): #anomaly比較圖
    import numpy as np
    import matplotlib.pyplot as plt
    #encoded_imgs = encoder.predict(x_test)
    #decoded_imgs = decoder.predict(encoded_imgs)
    predictions=autoencoder.predict(features)    
    mse=np.mean(np.power(features-predictions,2),axis=1)
    if mse.ndim>2:
        mse=np.sum(mse,axis=1)
        mse=(np.squeeze(mse)) 
    n=sum(mse>outlierMse) #how many digits we will display
    plt.figure(figsize=(n,2))
    j=0
    for i in range(len(mse)):
        if mse[i]>outlierMse:
            ax = plt.subplot(2, n, j + 1)
            plt.imshow(features[i].reshape(zoom, zoom))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)            
            az = plt.subplot(2, n, j + 1 + n)
            plt.imshow(predictions[i].reshape(zoom, zoom))
            plt.gray()
            az.get_xaxis().set_visible(False)
            az.get_yaxis().set_visible(False)
            j=j+1
    
def amonalyConfusionMatrix(targetLabel,filesLabelName,features,autoencoder,outlierMse):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    label_desc = [ 'Normal','Abnormal'] #類別資訊
    predictions=autoencoder.predict(features)
    mse=np.mean(np.power(features-predictions,2),axis=1)
    if mse.ndim>2:
        mse=np.sum(mse,axis=1)
        mse=(np.squeeze(mse))   
    predTargetlabelBooling=mse>outlierMse
    targetlabelBooling=(filesLabelName["Label"]!=targetLabel)    
    #checkList = pd.DataFrame( {'label':targetlabelBooling,'prediction':predTargetlabelBooling})
    cnf_matrix = confusion_matrix(targetlabelBooling, predTargetlabelBooling)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.array(label_desc),title='Confusion Matrix, Anomaly Dection')
    
def plotClass(features,label):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="ticks",color_codes=True)
    df = pd.DataFrame(features)
    df['label']=label
    fg = sns.pairplot(data=df, hue='label')
    fg.map(plt.scatter, 'good', 'bad').add_legend()
    plt.show()
