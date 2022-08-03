# -*- coding: utf-8 -*-
"""
my_model
HAMFD
"""

'''
gpu test
'''
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# print(tf.__version__)
# print('GPU',tf.test.is_gpu_available())
# model = Sequential()


'''
import
'''
# import sklearn
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# import tensorflow as tf
# import tensorflow.keras as keras
import keras
# from tcn import TCN, tcn_full_summary

from pyecharts import options as opts
from pyecharts.charts import HeatMap

import seaborn as sns

random.seed(10)


'''
data hierarchy
'''
group1_1=[2,4,84,133,174,115]
group1_2=[131,132,134,88]
group1_3=[135,136,137,138,139]
group1_4=[145,146,147,148]
group1_5=[140,141,142,143,144]
group1_6=[5,8,12,15,31,44,53,74,85,93,168]
group1_7=[0,1,175,2,103,86,157,164,178,190,191]
group1_8=[6,7,45,57,61,65,69,73,76,78,80,82,84]
group1_9=[89,94,97,100]
group1_10=[95,98,101]
group1_11=[96,99,102]
group1_12=[169,170,171,172,173,179,180,181]
group1_13=[182,183]
group1_14=[184,185,186]
group1_15=[187,188,189]
group1_16=[54,58,62,66,70]
group1_17=[55,59,63,67,71]
group1_18=[56,60,64,68,72]
group1_19=[75,77,79,81,83]
group1_20=[116,117,118]
group1_21=[119,120]
group1_22=[21,22,23,24,25,26,27,28,29,30]
group1_23=[32,33,34,35,36,37,38,39,40,41,42,43,163]
group1_24=[90,91,92]
group1_25=[9,10,11]
group1_26=[13,14]
group1_27=[16,17,18,19,20]
group1_28=[149,150,154,155]
group1_29=[46,47,48,49,50,51,52]
group1_30=[165,166,167]
group1_31=[151,152,153,104,105,156,106,107,108,109,110,111,112,113,114,121,122,123,124,125,126,127,128,129,130,162,176,158,159,160,161]

group1=[]
for i in range(31):
    exec('group1.append(group1_%d)'%(i+1))

group2_1=[1]
group2_2=[2]
group2_3=[3,4,5]
group2_4=[6,7]
group2_5=[8]
group2_6=[9,10,11]
group2_7=[12,13,14,15]
group2_8=[16,17,18,19,20,21]
group2_9=[22,23,24,25,26,27,28,29,30]
group2_10=[31]

group2=[]
for i in range(10):
    exec('group2.append(group2_%d)'%(i+1))

group3_1=[1,2,3,4,5,6,7,8,9,10]
group3=[]
group3.append(group3_1)

'''
data: raw01
sw_size: 60
sw_step: 5
mid: 3194
frep: 0.4
'''

raw=pd.read_csv('./data/raw01.csv').drop(['time'],axis=1)

data=[]
for i in group1:
    data.append(raw.iloc[:,i])

##滑窗
sw_size=60
sw_step=5
data_sw=[]
for i in range(len(data)):
    data_sw.append([data[i].iloc[j:j+sw_size,:].T for j in range(0,data[1].shape[0]-sw_size,sw_step)])
    
    
##label
mid=83194

mid=int(mid/sw_step)
data_y=list(np.ones(mid))+list(np.zeros(len(data_sw[0])-mid))


##positive_index and negivative index
p_index=list(range(mid))
n_index=list(range(mid,len(data_sw[0])))


##test set
frep=0.4

test_p_index=random.sample(p_index,int(frep*len(p_index)))
test_n_index=random.sample(n_index,int(frep*len(n_index)))

##train set
train_p_index=list(set(p_index)^set(test_p_index))
train_n_index=list(set(n_index)^set(test_n_index))

train_data=data_sw.copy()
test_data=data_sw.copy()

train_y=data_y.copy()
test_y=data_y.copy()


if len(train_p_index)/len(train_n_index)>1:    
    a=train_p_index+train_n_index*int(len(train_p_index)/len(train_n_index))
else:
    a=train_p_index*int(len(train_n_index)/len(train_p_index))+train_n_index

##shuffle
random.shuffle(a)
##train_data and train_y
train_y = type(train_y)(map(lambda i:train_y[i], a))
for i in range(31):
    train_data[i]=type(train_data[i])(map(lambda j:train_data[i][j], a))


a=test_p_index+test_n_index
##test_data and test_y
test_y = type(test_y)(map(lambda i:test_y[i], a))
for i in range(31):
    test_data[i]=type(test_data[i])(map(lambda j:test_data[i][j], a))



'''
model
'''

##softmax
def softmax(x):
    return keras.backend.softmax(x, axis=1)

##one_step_of_attention
def one_step_of_attention(h_prev, a):
    x = keras.layers.RepeatVector(keras.backend.int_shape(a)[1])(h_prev)
    # x=keras.backend.repeat(h_prev,K.int_shape(a)[1])
    x = keras.layers.Concatenate(axis=-1)([a, x])
    # x=keras.layers.concatenate([a,x],axis=-1)   
    x = keras.layers.Dense(8, activation="tanh")(x)
    x = keras.layers.Dense(1, activation="relu")(x)
    x = keras.layers.Activation(softmax)(x)
    attention = x
    x = keras.layers.Dot(axes=1)([attention, a])
    return x,attention



##attention_layer with many one_step_of_attention
def attention_layer(X, n_h, Ty):
    # h=keras.layers.Lambda(zeros, arguments={'n_h':n_h})(X)
    h = keras.layers.Lambda(lambda X: keras.backend.zeros(shape=(keras.backend.shape(X)[0], n_h)))(X)
    # keras.backend.shape(X)[0]
    # h =keras.backend.zeros(shape=(keras.backend.shape(X)[0], n_h))
    c = h   
    at_LSTM = keras.layers.LSTM(n_h, return_state=True)  
    output = []            
    for _ in range(Ty):
        context,attention = one_step_of_attention(h, X)       
        h, _, c = at_LSTM(context, initial_state=[h, c])
        output.append(h)
    return output,attention

##slice_index
def slice_index(x,index):
    return x[:, :, index]

##slice_all
def slice_all(x):
    value=[]
    for j in range(keras.backend.int_shape(x)[2]):
        value.append(keras.layers.Lambda(slice_index, arguments={'index':j})(x))
    return value


##orth_cal
def orth_cal(aver,old):
    dot1=keras.layers.Dot(axes=1)([aver,old])
    # dot1=keras.backend.batch_dot(aver,old)
    dot2=keras.layers.Dot(axes=1)([old,old])
    # dot2=keras.backend.batch_dot(old,old)
    # Schmidt quadrature
    div=keras.layers.Lambda(lambda inputs: inputs[0]/(inputs[1])+1e-12)([dot1,dot2])
    # div=dot1/dot2
    repeat=keras.layers.Lambda(slice_index, arguments={'index':0})(keras.layers.RepeatVector(keras.backend.int_shape(old)[1])(div))
    # repeat=keras.backend.repeat(div,keras.backend.int_shape(old)[1])[:,:,0]
    mul=keras.layers.Multiply()([repeat,old])
    # mul=keras.layers.multiply([repeat,old])
    sub=keras.layers.Subtract()([aver,mul])
    # sub=keras.layers.subtract([aver,mul]) 
    
    return sub

##orth_layer_first
def orth_layer_first(X):
    return X

##orth_layer_second
def orth_layer_second(X):
    value=[]
    for j in range(keras.backend.int_shape(X)[2]):
        value.append(keras.layers.Lambda(slice_index, arguments={'index':j})(X))
        # value.append(X[:,:,j])
    # value=keras.layers.Lambda(slice_all)(X)
    
    aver=value[0]
    old=value[1]

    # sub=orth_cal(aver,old)

    dot1=keras.layers.Dot(axes=1)([aver,old])
    # dot1=keras.backend.batch_dot(aver,old)
    dot2=keras.layers.Dot(axes=1)([old,old])
    # dot2=keras.backend.batch_dot(old,old)
    div=keras.layers.Lambda(lambda inputs: inputs[0]/(inputs[1]+1e-12))([dot1,dot2])
    # div=dot1/dot2
    repeat=keras.layers.Lambda(slice_index, arguments={'index':0})(keras.layers.RepeatVector(keras.backend.int_shape(old)[1])(div))
    # repeat=keras.backend.repeat(div,keras.backend.int_shape(old)[1])[:,:,0]
    mul=keras.layers.Multiply()([repeat,old])
    # mul=keras.layers.multiply([repeat,old])
    sub=keras.layers.Subtract()([aver,mul])
    # sub=keras.layers.subtract([aver,mul]) 
    
    value[1]=sub
    
    con=keras.layers.Concatenate()(value)
    # con=keras.layers.concatenate(value,axis=-1)
    res=keras.layers.Reshape((keras.backend.int_shape(X)[1],keras.backend.int_shape(X)[2]))(con)
    # res=keras.backend.reshape(con, (keras.backend.int_shape(X)[1],keras.backend.int_shape(X)[2]))
    
    return res

##orth_layer_third
def orth_layer_third(X,n):
    value=[]
    for j in range(keras.backend.int_shape(X)[2]):
        value.append(keras.layers.Lambda(slice_index, arguments={'index':j})(X))
    aver=keras.layers.Average()(value[0:n])
    # aver=keras.layers.average(value[0:n])
    old=value[n]
    # sub=orth_cal(aver,old)
    
    
    dot1=keras.layers.Dot(axes=1)([aver,old])
    # dot1=keras.backend.batch_dot(aver,old)
    dot2=keras.layers.Dot(axes=1)([old,old])
    # dot2=keras.backend.batch_dot(old,old)
    div=keras.layers.Lambda(lambda inputs: inputs[0]/(inputs[1]+1e-12))([dot1,dot2])
    # div=dot1/dot2
    repeat=keras.layers.Lambda(slice_index, arguments={'index':0})(keras.layers.RepeatVector(keras.backend.int_shape(old)[1])(div))
    # repeat=keras.backend.repeat(div,keras.backend.int_shape(old)[1])[:,:,0]
    mul=keras.layers.Multiply()([repeat,old])
    # mul=keras.layers.multiply([repeat,old])
    sub=keras.layers.Subtract()([aver,mul])
    # sub=keras.layers.subtract([aver,mul]) 
    
    value[n]=sub
    
    con=keras.layers.Concatenate()(value)
    # con=keras.layers.concatenate(value,axis=-1)
    res=keras.layers.Reshape((keras.backend.int_shape(X)[1],keras.backend.int_shape(X)[2]))(con)
    # res=keras.backend.reshape(con, (keras.backend.int_shape(X)[1],keras.backend.int_shape(X)[2]))
    
    return res

##orth_layer
def orth_layer(X):
    # X=orth_layer_first(X)
    # X=orth_layer_second(X)
    # for n in range(2,keras.backend.int_shape(X)[2]):
        # X=orth_layer_third(X,n)
    return X

##define the model
##keras api
# batch_size, timesteps, input_dim = None, 20, 1

##first_layer_model
input1=[]
lstm1=[]
orth1=[]
attention1_layer=[]
attention1=[]
model1=[]
normal=[]

for i in range(len(group1)):
    # print(i)
    input1.append(keras.layers.Input(batch_shape=(None,len(group1[i]),sw_size)))      
    normal.append(keras.layers.BatchNormalization(axis=-1)(input1[i]))
    # lstm1.append(TCN(64,return_sequences=True)(normal[i]))       
    lstm1.append(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='concat')(normal[i]))
    
    orth1.append(keras.layers.Lambda(orth_layer)(lstm1[i]))    
    # orth.append(orth_layer(lstm1[i]))
    
    attention1_output1,attention1_output2=attention_layer(orth1[i], 64, 1)
    
    attention1_layer.append(attention1_output1) 
    attention1.append(attention1_output2)            
    # attention1.append(attention_layer(lstm1[i], 64, 1))
    
    model1.append(keras.models.Model(inputs=[input1[i]], outputs=attention1_layer[i]))



##Layer 1 Model Aggregation
comb1=[]
for i in range(len(group2)):
    if len(group2[i])==1:
        comb1.append([])
    else:
        temp_output=[]
        for j in group2[i]:
            temp_output.append(model1[j-1].output)
        comb1.append(keras.layers.Reshape((len(group2[i]),64))(keras.layers.Concatenate()(temp_output)))
      
##Layer 2 Model
attention2_layer=[]
attention2=[]
model2=[]

for i in range(len(group2)):
    if len(group2[i])==1:
        attention2_layer.append([])
        attention2.append([])
        model2.append(model1[group2[i][0]-1])
    else:
        
        attention2_output1,attention2_output2=attention_layer(comb1[i], 64, 1)
        attention2_layer.append(attention2_output1)
        attention2.append(attention2_output2)
        temp_input=[]
        for j in group2[i]:
            temp_input.append(model1[j-1].input)
        model2.append(keras.models.Model(inputs=temp_input, outputs=attention2_layer[i]))  

##Layer 2 Model Aggregation
comb2=[]
temp_output=[]
for j in group3[0]:
    temp_output.append(model2[j-1].output)
comb2.append(keras.layers.Reshape((len(group3[0]),64))(keras.layers.Concatenate()(temp_output)))

##Layer 3 Model and mhat
attention3_layer=[]
attention3=[]
model3=[]
model3_attention=[]

for i in range(8):
    attention3_output1,attention3_output2=attention_layer(comb2[0], 64, 1)
    attention3_layer.append(attention3_output1)
    attention3.append(attention3_output2)
    temp_input=[]
    for j in range(len(group1)):
        temp_input.append(model1[j].input) 
    model3.append(keras.models.Model(inputs=temp_input,outputs=attention3_layer[i]))

##Layer 3 Model Aggregation
comb3=[]
temp_output=[]
for i in range(8):
    temp_output.append(model3[i].output)
comb3.append(keras.layers.Reshape((8,64))(keras.layers.Concatenate()(temp_output)))

attention4_layer,attention4=attention_layer(comb3[0], 64, 1)

##mlp
dense1 = keras.layers.Dense(32, activation="relu")(attention4_layer[0])
dense2 = keras.layers.Dense(8, activation="relu")(dense1)
dense3 = keras.layers.Dense(2, activation="sigmoid")(dense2)

##whole model
model=keras.models.Model(inputs=model3[0].input,outputs=dense3)

'''
save,load,compile,optimizer,lr,train
'''

# model.save('init.h5')
# model=keras.models.load_model('init.h5')
# model.save_weights('weight/init_weight.h5')
# model.save_weights('weight/trained_weight_59.h5')
# model.load_weights('weight/init_weight_56_1.h5.h5')

##plot_model
keras.utils.plot_model(model1[1],to_file='_model_.png',show_shapes=True)

# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return optimizer.lr
#     return lr
# lr_metric = get_lr_metric(optimizer)

# optimizer = keras.optimizers.Adam(lr=0.001, clipvalue=0.0001)
optimizer = keras.optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

# history=model.fit(data_sw, data_y, epochs=2, batch_size=32,callbacks=[lr])
# history=model.fit(data_sw, data_y, epochs=30, batch_size=32,shuffle=True,class_weight={1:1,0:21})

train_y_onehot = keras.utils.to_categorical(train_y)
test_y_onehot = keras.utils.to_categorical(test_y)
history=model.fit(train_data,train_y_onehot,validation_data=(test_data, test_y_onehot),epochs=5, batch_size=32)

'''
performance
'''

##learning curve
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])

prepro=np.round(model.predict(data_sw))
pre=1-(prepro[:,0]>prepro[:,1])

data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=metrics.f1_score(data_y2, pre2)
cm=metrics.confusion_matrix(data_y2, pre2)


'''
attention weight
'''

model1_attention=[]
for i in range(len(group1)):
    model1_attention.append(keras.models.Model(inputs=model3[0].input, outputs=attention1[i]))
    
model2_attention=[]
for i in range(len(group2)):
    if len(group2[i])==1:
        model2_attention.append([])
    else:
        model2_attention.append(keras.models.Model(inputs=model3[0].input, outputs=attention2[i]))    
    
model3_attention=[]
for i in range(8):
    model3_attention.append(keras.models.Model(inputs=model3[0].input, outputs=attention3[i]))  

model4_attention=keras.models.Model(inputs=model3[0].input,outputs=attention4)

    
attention1_output=[]
for i in range(len(group1)):
    attention1_output.append(model1_attention[i].predict(data_sw))
    
attention2_output=[]
for i in range(len(group2)):
    if len(group2[i])==1:
        attention2_output.append(np.ones((len(data_sw[0]),1,1)))
    else:
        attention2_output.append(model2_attention[i].predict(data_sw))
    
attention3_output=[]
for i in range(8):
    attention3_output.append(model3_attention[i].predict(data_sw))

attention4_output=model4_attention.predict(data_sw)

##attention_output4
for i in range(len(attention4_output)):
    if i==0:
        attention_output4=attention4_output[i].T
    else:
        attention_output4=np.r_[attention_output4,attention4_output[i].T]
        
##attention_output3
attention_output3=[]
for i in range(len(attention3_output)):
    for j in range(len(attention3_output[0])):
        if j==0:
            temp=attention3_output[i][j].T  
        else:
            temp=np.r_[temp,attention3_output[i][j].T]   
    attention_output3.append(temp)   

##attention_output2
attention_output2=[]
for i in range(len(attention2_output)):
    for j in range(len(attention2_output[0])):
        if j==0:
            temp=attention2_output[i][j].T  
        else:
            temp=np.r_[temp,attention2_output[i][j].T]   
    attention_output2.append(temp)   

##attention_output1
attention_output1=[]
for i in range(len(attention1_output)):
    for j in range(len(attention1_output[0])):
        if j==0:
            temp=attention1_output[i][j].T  
        else:
            temp=np.r_[temp,attention1_output[i][j].T]   
    attention_output1.append(temp)   

##importanmce of layer 3
temp=attention_output3.copy()
for i in range(attention_output4.shape[1]):
    for j in range(attention_output4.shape[0]):
        temp[i][j] = attention_output4[j,i]*temp[i][j]
for i in range(len(temp)):
    if i==0:
        attention_group2=temp[i]  
    else:
        attention_group2=attention_group2+temp[i]  
          
            

##importanmce of layer 2
temp=attention_output2.copy()
for i in range(attention_group2.shape[1]):
    for j in range(attention_group2.shape[0]):
        temp[i][j] = attention_group2[j,i]*temp[i][j]
for i in range(len(temp)):
    if i==0:
        attention_group1=temp[i]
    else:
        attention_group1=np.c_[attention_group1,temp[i]]
        

##importanmce of layer 1
temp=attention_output1.copy()
for i in range(attention_group1.shape[1]):
    for j in range(attention_group1.shape[0]):
        temp[i][j] = attention_group1[j,i]*temp[i][j]
for i in range(len(temp)):
    if i==0:
        attention_group0=temp[i]
    else:
        attention_group0=np.c_[attention_group0,temp[i]]

##visualization

##heatmap

plt.figure()
sns.heatmap(attention_group2, vmin=0, vmax=0.5, cmap="Blues",cbar=True)#RdPu,Blues,Blues_r,BrBG
plt.figure()
sns.heatmap(attention_group1, vmin=0, vmax=0.5, cmap="Blues",cbar=True)
plt.figure()
sns.heatmap(attention_group0, vmin=0, vmax=0.1, cmap="Blues",cbar=True)

##pyecharts

pycharts_data=attention_group0.copy()

value = [[j, i, pycharts_data[i,j]*100] for i in range(pycharts_data.shape[0]) for j in range(pycharts_data.shape[1])]
c = (
    HeatMap(init_opts=opts.InitOpts(height="900px",width='1500px'))
    .add_xaxis(list(range(pycharts_data.shape[1])))
    .add_yaxis("attention_group0", list(range(pycharts_data.shape[0])), value)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="HeatMap-attention"),
        visualmap_opts=opts.VisualMapOpts(min_=0,max_=10),
        tooltip_opts=opts.TooltipOpts(axis_pointer_type="cross"),# 指示器类型
        # toolbox_opts =opts.ToolboxOpts(is_show=True,orient='vertical'),
        # toolbox_opts=opts.ToolBoxFeatureDataZoomOpts(is_show=True),
        datazoom_opts=opts.DataZoomOpts(range_start=0,range_end=int(pycharts_data.shape[1]/10),orient='vertical'), # 坐标轴进行缩放
    )
    .render("./html/heatmap_base.html")
)