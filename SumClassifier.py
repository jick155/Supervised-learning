# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:34:10 2018

@author: ST
"""

# -*- coding: utf-8 -*-

import dataSetIO
import resultPlot
import classifierModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.models import load_model

#---------相關路徑設定---------
trainPicPath="C:/data/PCB_DEFECT/train_images" #訓練圖片資料路徑
testPicPath="C:/data/pcbDefect/test_images"    #測試圖片資料路徑
trainlabelPath="C:/data/PCB_DEFECT/train.csv"  #訓練label路徑
outpicFilePath = "C:/data/PCB_DEFECT/"         #輸出圖片資料路徑
tablePath="C:/data/pcbDefect/letNet5/overSampling/ecoph1000/leNet1000.csv" #儲存CSV檔路徑&檔案名稱
colname="label" #對應寫入欄位名
testLabelPath="C:/data/pcbDefect/test.csv"  #測試圖片資料路徑
modelPath="C:/data/pcbDefect/letNet5/overSampling/ecoph1000/AOI_letNet5_OverSampling1000_Adam_Batch200_.h5" #儲存Model路徑
#---------相關路徑設定---------

#---------相關參數設定---------
zoom=48 #縮放大小
input_shape=(zoom,zoom,1) #輸入shape
ylabel=np.array(pd.read_csv(trainlabelPath)['Label']) #訓練資料對應圖片類別
yPicName=pd.read_csv(trainlabelPath)['ID'] #訓練資料對應圖片名稱
picFormat="png" #圖片檔案格式
label_desc = [ '0','1', '2', '3', '4', '5'] #類別資訊
classes=len(label_desc) #類別總數
indexList=range(0, 10)  # 測試結果顯示資料起始與結束位置
targetLabel=0 #目標類別
testSampleQty=10 #測試數量
#---------相關參數設定---------


#---------#資料準備--------- 分為測試死訓練集
x=dataSetIO.picToMat(trainPicPath,zoom,picFormat) #輸入資料

#test_index=np.random.choice(range(0,x.shape[0]),testSampleQty,replace=False) #測試指標
#train_index=np.delete(np.array(range(0,x.shape[0])),test_index) #訓練指標
#train_feature=x[np.array(train_index),:,:,np.newaxis] #訓練圖片資料
#train_label=ylabel[np.array(train_index)] #訓練類別
#test_feature=x[np.array(test_index),:,:,np.newaxis] #測試圖片資料
#test_label=ylabel[np.array(test_index)] #測試類別

train_feature,train_label,train_yPicName,test_feature,test_label,test_yPicName=dataSetIO.trainTestSet(xdata=x[:,:,:,np.newaxis],ylabel=ylabel,yPicName=yPicName,test_percentage=0.01) #輸出資料對應的類別與資料名稱
train_label_onehot = np_utils.to_categorical(train_label) #訓練類別轉換one_hot_encoding

#---------資料準備---------



#---------Moldel與內部參數選擇---------

#####LetNet5#########測試pass
model=classifierModel.lenet_5(classes,zoom)

#####VGG16######### zoom 使用 weights='imagenet'至少 48 #測試pass
#model=classifierModel.VGG16(classes,zoom)

#####VGG19######### zoom 使用 weights='imagenet'至少 48 #測試pass
#model=classifierModel.VGG19(classes,zoom)

#####InceptionV3######### zoom 至少 139 #測試pass
#model=classifierModel.InceptionV3(classes,zoom)

#####ResNet50####### zoom 至少 197 #測試pass
#train_feature=dataSetIO.GrayTranfColor(train_feature,zoom) #訓練資料灰階1通道轉乘彩色3通道
#test_feature=dataSetIO.GrayTranfColor(test_feature,zoom) #測試資料灰階1通道轉乘彩色3通道
#model=classifierModel.ResNet50(classes,zoom)

######InceptionResNetV2######
#model=classifierModel.InceptionResNetV2(classes,zoom)

#####MnasNet#########測試 pass
#alpha=1.0
#depth_multiplier=1
#pooling=None
#model=classifierModel.MnasNet(alpha, depth_multiplier, pooling, classes,zoom)

#---------Moldel決定---------


#---------Model 編譯與外部參數選擇---------
model.compile( loss='categorical_crossentropy' # 設定 Loss 損失函數 為 categorical_crossentropy
             , optimizer = 'adam'              # 設定 Optimizer 最佳化方法 為 adam
             , metrics = ['accuracy']          # 設定 Model 評估準確率方法 為 accuracy
             )

history = model.fit(               # 訓練的歷史記錄, 會會回傳到指定變數 history
          x = train_feature        # 設定 圖片 Features 特徵值 
        , y = train_label_onehot   # 設定 圖片 Label    真實值 
        , validation_split = 0.2   # 設定 有多少筆驗證         
        , epochs = 100              # 設定 訓練次數             (值 10 以上,  值越大, 訓練時間越久, 但訓練越精準)
        , batch_size = 10        # 設定 訓練時每批次有多少筆 (值 100 以上, 值越大, 訓練速度越快, 但需記憶體要夠大,記憶體不夠時可調小)
        , verbose = 2              # 是否 顯示訓練過程         (0: 不顯示, 1: 詳細顯示, 2: 簡易顯示)
)
#---------Model 編譯與外部參數選擇---------


#---------訓練結果畫圖---------
resultPlot.train_history_graphic( history, 'acc', 'val_acc', 'accuracy') #畫正確率圖
resultPlot.train_history_graphic( history, 'loss', 'val_loss', 'loss' ) #畫loss圖
#---------訓練結果畫圖---------

#---------測試資料預測結果 與 confusion_matrix---------
prediction = np.argmax(model.predict(test_feature),axis=1)
checkList = pd.DataFrame( {'label':test_label,'prediction':prediction})
cnf_matrix = confusion_matrix(test_label, prediction)
np.set_printoptions(precision=2)
plt.figure()
resultPlot.plot_confusion_matrix(cnf_matrix, classes=np.array(label_desc),title='Confusion matrix, without add sample')
#---------測試資料預測結果 與 confusion_matrix---------

#---------測試資料預測結果圖案---------
resultPlot.show_feature_label_prediction( test_feature, test_label, prediction, range(0, 10),label_desc) #畫出所有圖案
resultPlot.show_feature_label_prediction( test_feature, test_label, prediction, 
checkList.index[checkList.prediction != checkList.label][0:10],label_desc) #畫出判錯的圖案
#---------測試資料預測結果圖案---------

model.save('C://python37//program//fineTuneModels//vgg19_Adam_Batch200.h5')   #存訓練參數
model = load_model('C://python37//program//fineTuneModels//vgg19_Adam_Batch200.h5') #讀訓練參數
