# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:40:39 2018

@author: ST
"""

def imgGenByImageDataGenerator(imgNdarray,imgLabel,TargetLabel,genQty,genPam): #傳統圖片生成(偏移,旋轉,放大縮小)
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    #相關參數 網址:https://keras.io/zh/preprocessing/image/
    img_generator = ImageDataGenerator(
        rotation_range=genPam[0], #隨機旋轉角度範圍/整數
        width_shift_range=genPam[1], #圖像x軸偏移/浮點數
        height_shift_range=genPam[2], #圖像y軸偏移/浮點數
        shear_range=genPam[3], #以弧度逆時針剪切角度/浮點數
        zoom_range=genPam[4], #放大縮小/整數 or [lower, upper]
        horizontal_flip=genPam[5], #隨機守平轉 布爾值
        fill_mode=genPam[6] #ex:'nearest'
        )    
    x = imgNdarray[imgLabel==TargetLabel,:,:,:]     #選出要增加照片類別的所有圖片       
    imgGenNdarray=np.expand_dims(x[0], axis=0)      #生成图片 
    for i in range(x.shape[0]):
        gen = img_generator.flow(np.expand_dims(x[i], axis=0), batch_size=1)
        for j in range(genQty):
            x_batch = next(gen)
            imgGenNdarray=np.vstack((imgGenNdarray,x_batch))
            
    imgGenNdarray=imgGenNdarray[1:imgGenNdarray.shape[0],:,:,:] #imgGenNdarray:生成照片矩陣
    imgGenLabel=(np.array(2).repeat(imgGenNdarray.shape[0]))    #imgGenLabel:生成照片標籤
    return imgGenNdarray,imgGenLabel


def imgGenBySame(imgNdarray,imgLabel,TargetLabel,genQty): #相同照片增加
    import numpy as np   
    x = imgNdarray[imgLabel==TargetLabel,:,:,:]     #選出要增加照片類別的所有圖片       
    imgGenNdarray=np.expand_dims(x[0], axis=0)      #生成图片 
    for i in range(genQty-1):
            imgGenNdarray=np.vstack((imgGenNdarray,x))
            
    imgGenNdarray=imgGenNdarray[1:imgGenNdarray.shape[0],:,:,:] #imgGenNdarray:生成照片矩陣
    imgGenLabel=(np.array(2).repeat(imgGenNdarray.shape[0]))    #imgGenLabel:生成照片標籤
    return imgGenNdarray,imgGenLabel


#####gan#####