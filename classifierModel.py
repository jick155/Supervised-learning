# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:37:25 2018

@author: ST
"""

def lenet_5(classes,zoom):
    from keras.layers import Input,Conv2D,Dense,ZeroPadding2D,Activation,MaxPooling2D,Flatten
    from keras.models import Model
    input_shape=(zoom,zoom,1)
    X_input=Input(input_shape)
    X=ZeroPadding2D((1,1))(X_input)
    X=Conv2D(130,(5,5),strides=(1,1),padding='same',name='conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((3,3),strides=(2,2))(X)
    X=Conv2D(60,(3,3),strides=(1,1),padding='same',name='conv2')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),strides=(2,2))(X)
    X=Flatten()(X)
    X=Dense(120,activation='relu',name='fc1')(X)
    X=Dense(84,activation='relu',name='fc2')(X)
    X=Dense(6,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X,name='lenet_5')
    return model

def VGG16(classes,zoom):
    from keras.layers import Dense,Flatten
    from keras.models import Model
    from keras.applications.vgg16 import VGG16
    model_vgg=VGG16(include_top=False,weights=None,input_shape=(zoom,zoom,1))
    for layer in model_vgg.layers:
        layer.trainable=False
    model=Flatten(name='Flatten')(model_vgg.output)
    model=Dense(classes,activation='softmax')(model)
    model=Model(model_vgg.input,model,name='vgg16')
    return model

def VGG19(classes,zoom):
    from keras.layers import Dense,Flatten
    from keras.models import Model
    from keras.applications.vgg19 import VGG19
    model_vgg=VGG19(include_top=False,weights=None,input_shape=(zoom,zoom,1))
    for layer in model_vgg.layers:
        layer.trainable=False
    model=Flatten(name='Flatten')(model_vgg.output)
    model=Dense(classes,activation='softmax')(model)
    model=Model(model_vgg.input,model,name='vgg19')
    return model

def InceptionV3(classes,zoom):
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    model_InceptionV3 = InceptionV3(weights=None, include_top=False,input_shape=(zoom, zoom, 1))
    x = model_InceptionV3.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(model_InceptionV3.input, outputs=predictions)
    return model
    
def ResNet50(classes,zoom):
    from keras.layers import Dense,Flatten
    from keras.models import Model
    from keras.applications.resnet50 import ResNet50
    ResNet50(include_top=False,weights=None,input_shape=(zoom,zoom,3))
    model_ResNet50=ResNet50(include_top=False,weights='imagenet',input_shape=(zoom,zoom,3))
    for layer in model_ResNet50.layers:
        layer.trainable=False
        model=Flatten(name='Flatten')(model_ResNet50.output)
        model=Dense(classes,activation='softmax')(model)
        model=Model(model_ResNet50.input,model,name='ResNet50')
    return model

def InceptionResNetV2(classes,zoom):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    model_InceptionResNetV2 = InceptionResNetV2(include_top=False, weights=None, input_shape=(zoom,zoom,1))
    x = model_InceptionResNetV2.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(model_InceptionResNetV2.input, outputs=predictions)
    return model

def MnasNet(alpha, depth_multiplier, pooling, classes, zoom):
    
    from keras import layers
    from keras import models
    import classifierMnasNet
    
    input_shape=(zoom,zoom,1)
    
    img_input = layers.Input(shape=input_shape)
    
    first_block_filters = classifierMnasNet._make_divisible(32 * alpha, 8)
    x = classifierMnasNet._conv_block(img_input, strides=2, filters=first_block_filters)

    x = classifierMnasNet._sep_conv_block(x, filters=16, alpha=alpha, pointwise_conv_filters=16, depth_multiplier=depth_multiplier)
    
    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=3, stride=2, alpha=alpha, filters=24, block_id=1)
    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=2)
    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=3)
    
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=3, stride=2, alpha=alpha, filters=40, block_id=4)

    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=5)
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=6)
    
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=80, block_id=7)

    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=8)
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=9)

    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=10)
    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=11)
    
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=12)

    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=13)
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=14)
    x = classifierMnasNet._inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=15)
    
    x = classifierMnasNet._inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=320, block_id=16)
    
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)
        
    x = layers.Dense(classes, activation='softmax', use_bias=True, name='proba')(x)
    
    inputs = img_input
    
    model = models.Model(inputs, x, name='mnasnet')
    return model

def firstone(classes,zoom):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    import keras.layers
    model = Sequential()
    model.add( Conv2D( input_shape = (zoom, zoom, 1)   
                     , filters = 8                   
                     , kernel_size = (3, 3)          
                     , padding = 'same'              
                     , activation = 'relu'           
                     )
             )
    model.add( MaxPooling2D( pool_size = (2, 2) ) )    
    model.add( Conv2D( input_shape = (zoom, zoom, 1)     
                 , filters = 12                 
                 , kernel_size = (3, 3)          
                 , padding = 'same'              
                 , activation = 'relu'          
                 )
         )
    model.add( MaxPooling2D( pool_size = (3, 3) ) )                                                 
    model.add( Conv2D( input_shape = (zoom, zoom, 1)    
                 , filters = 12                  
                                                 
                 , kernel_size = (3, 3)          
                 , padding = 'same'             
                 , activation = 'relu'         
                 )
         )
    model.add( MaxPooling2D( pool_size = (3, 3) ) )                                                       
    model.add( Conv2D( input_shape = (zoom, zoom, 1)    
                 , filters = 12                  
                 , kernel_size = (3, 3)          
                 , padding = 'same'             
                 , activation = 'relu'          
                 )
         )
    model.add( MaxPooling2D( pool_size = (3, 3) ) )
    model.add( Flatten() )
    #act=keras.layers.ReLU(max_value=None)
    act=keras.layers.Softmax(axis=-1)
    #act=keras.layers.ThresholdedReLU(theta=0.5)
    #act=keras.layers.ELU(alpha=1.0)
    #act=keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
    #act=keras.layers.LeakyReLU(alpha=0.3)
    model.add(act)
    model.add( Dense( units = 256                   # 隱藏層有 256 個神經元 (值越大, 訓練越精準, 相對訓練時間也越久)
                    , kernel_initializer = 'normal' # 使用 normal 初始化 weight 權重與 bias 偏差值
                    , activation = 'relu'           # 使用 relu 激活函數
                    )
             )
    model.add( Dense( units = 6                    # 輸出層有 10 個神經元 (因為數字只有 0 ~ 9)
                    , kernel_initializer = 'normal' # 使用 normal 初始化 weight 權重與 bias 偏差值
                    , activation = 'softmax'        # 使用 softmax 激活函數 (softmax 值越高, 代表機率越大)
                    )
             )
    return model