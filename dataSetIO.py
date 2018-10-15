# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:53:48 2018

@author: ST
"""

#class DataSetIO():
#    def __init__(self,picPath,zoom,picFormat):
#        self.picPath=picPath
#        self.zoom=zoom
#        self.picFormat=picFormat
def picToMat(trainPicPath,zoom,picFormat): #圖片轉矩陣
    import glob,cv2
    import matplotlib.pyplot as plt
    import numpy as np
    files = glob.glob(trainPicPath + r'\*.'+picFormat,recursive=True)
    matrixs=np.stack([cv2.resize(plt.imread(file),(zoom,zoom)) for file in files]) 
    return matrixs

def selectLabelClass(xdata,ylabel,targetLabel,yPicName):  #選擇類別
    import numpy as np
    #xdata=xdata
    selectX=(xdata[:,:,:,np.newaxis])[np.squeeze(ylabel[:,np.newaxis]==targetLabel,axis=(1,)),:,:,:]
    selectY=(np.array(targetLabel).repeat(selectX.shape[0]))
    selectYPicName=(yPicName[np.squeeze(ylabel[:,np.newaxis]==targetLabel)]).reset_index(drop=True)
    return selectX,selectY,selectYPicName

def outPicFile(selectX,selectYPicName,targetLabel,outpicFilePath): #輸出目標圖片至資料夾
    import os,cv2
    import numpy as np
    dir_name = outpicFilePath+"class_%i"%targetLabel
    if not os.path.exists(dir_name):    #先確認資料夾是否存在
        os.makedirs(dir_name)    
    for i in range(len(selectX)):
            #im_gray = cv2.cvtColor(selectX[i], cv2.COLOR_BGR2GRAY) 彩色轉灰階
            cv2.imwrite(dir_name+"/"+selectYPicName[i],np.squeeze(selectX[i])*255)  

def predTable(tablePath,colname,testPicPath,modelPath,x): #輸出預測結果，存成csv檔
    from keras.models import load_model
    import pandas as pd
    import numpy as np
    model = load_model(modelPath)
    yPicName=(pd.read_csv(testPicPath)['ID']).tolist()  
    label=(np.argmax(np.round(model.predict(x[:,:,:,np.newaxis])),axis=1)).tolist()
    tableDict={"ID":yPicName,"Label":label}   
    df = pd.DataFrame(tableDict)
    df.to_csv(tablePath,index = False)
    
def GrayTranfColor(imgFeature,zoom):  #灰階1通道轉彩色3通道
    import cv2
    import numpy as np
    imgFeature=imgFeature.astype(np.float32) #使用cv2.cvtColor前須進行轉換步驟
    imgFeature=[cv2.cvtColor(cv2.resize(i,(zoom,zoom)),cv2.COLOR_GRAY2BGR) for i in imgFeature]#變成由灰階轉成RGB
    imgFeature=np.array(imgFeature)#轉成矩陣
    return imgFeature

def trainTestSet(xdata,ylabel,yPicName,test_percentage):   #產出訓練與測試資料對 "應標籤名稱" 與 "對應缺陷類別" 
    import numpy as np
    import pandas as pd
    import math
    test_index=np.random.choice(range(0,xdata.shape[0]),math.ceil(xdata.shape[0]*test_percentage),replace=False)   
    train_index=np.delete(np.array(range(0,xdata.shape[0])),test_index)
    
    train_feature=xdata[np.array(train_index),:,:,:]
    train_label=ylabel[np.array(train_index)] 
    #train_yPicName=((yPicName[np.array(train_index)]).dropna()).reindex(drop=True)
    train_yPicName=yPicName[np.array(train_index)]    
    train_yPicName=train_yPicName[pd.notnull(train_yPicName)]
    train_yPicName.index = range(len(train_yPicName))
    test_feature=xdata[np.array(test_index),:,:,:]
    test_label=ylabel[np.array(test_index)]
    #test_yPicName=((yPicName[np.array(test_index)]).dropna()).reindex(drop=True)
    test_yPicName=yPicName[np.array(test_index)]
    test_yPicName=test_yPicName[pd.notnull(test_yPicName)]
    test_yPicName.index = range(len(test_yPicName))
    return train_feature,train_label,train_yPicName,test_feature,test_label,test_yPicName

def outlierPicName(xdata,yPicName,autoencoder,outlierMse): #anomalyDection 輸出異常 mse 與 outlierPicName
    import pandas as pd
    import numpy as np
    predictions=autoencoder.predict(xdata)
    mse=np.mean(np.power(xdata-predictions,2),axis=1)
    if mse.ndim>2:
            mse=np.sum(mse,axis=1)
            mse=(np.squeeze(mse))
    outlierPicName=pd.Series()
    for i in range(len(mse)):
        if mse[i]>outlierMse:
            outlierPicName=pd.concat([outlierPicName,pd.Series(yPicName[i])],axis=0).reset_index(drop=True)   
    return mse,outlierPicName
    #outlierPicName['Index']=range(len(outlierPicName))

def getFileName(FilePath): #拿到檔案名稱
    from os import walk
    fileName = []
    #FilePath = 'C:\\data\\PCB_DEFECT\\autoencoderTestSet'
    for (dirpath, dirnames, filenames) in walk(FilePath):
        fileName.extend(filenames)
    return fileName

def filesPicLabelName(filesName,picLabelName): #檔案名稱與
    import pandas as pd
    filesLabelName =  pd.DataFrame()    
    for i in range(len(filesName)):
        fliter = (picLabelName["ID"] == filesName[i])        
        filesLabelName=pd.concat([filesLabelName,picLabelName[fliter]])
    return filesLabelName



#from keras.models import load_model
#from keras.models import Model
#model.save('C://python37//program//fineTuneModels//letNet5_Adam_Batch200.h5')
#model = load_model('C://python37//program//fineTuneModels//AOI_letNet5_Adam_Batch200.h5')