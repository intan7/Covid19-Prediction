# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:14:25 2022

@author: intan
"""
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.utils import plot_model

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from Covid19_Prediction_Module import ModelDevelopment,ModelEvaluation,PlotFig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import os
#%% PATH

CSV_PATH_TRAIN = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
MMS_PATH_X=os.path.join(os.getcwd(),'model','mms.pkl')
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
BEST_MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best_model.h5')
#%% Step 1)Data loading

df = pd.read_csv(CSV_PATH_TRAIN,na_values=[' ','?'])
df=df.interpolate(method='linear') #to fill for NaNs
df['cases_new']=df['cases_new'].round(0).astype(int)

df_test = pd.read_csv(CSV_PATH_TEST,na_values=[' ','?'])
df_test=df_test.interpolate(method='linear') #to fill for NaNs
df['cases_new']=df['cases_new'].round(0).astype(int)

#%% Step 2)Data inspection

df.head()
df.info()
df.isna().sum()
df.describe().T

df_disp=df[1:1000]
plt.figure()
plt.plot(df_disp['cases_new'])
plt.show()

#%% Step 3)Data cleaning
#%% Step 4)Features selection
#Train Dataset

X=df['cases_new']

mms=MinMaxScaler()
with open(MMS_PATH_X,'wb') as file:
    pickle.dump(mms,file)

X=mms.fit_transform(np.expand_dims(X,axis=-1))

win_size=30
X_train=[]
y_train=[]

for i in range(win_size, len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

X_train=np.array(X_train)
y_train=np.array(y_train)

#%% #Step 5)Data preprocessing
#Test Dataset

dataset_df=pd.concat((df['cases_new'],df_test['cases_new']))

length_data=len(dataset_df)-len(df_test)-win_size
tot_input=dataset_df[length_data:]

Xtest=mms.transform(np.expand_dims(tot_input,axis=-1))

X_test=[]
y_test=[]

for i in range(win_size, len(Xtest)):
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])

X_test=np.array(X_test)
y_test=np.array(y_test)

#%% Model Development

shape_x=np.shape(X_train)[1:]
output=1

md=ModelDevelopment()
model=md.simple_MD_model(shape_x,output,activation='linear',
                           nb_node=64,dropout_rate=0.1)

plot_model(model,show_shapes=(True))

#%% Model Training

model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mean_absolute_percentage_error','mse'])

#Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
mdc = ModelCheckpoint(BEST_MODEL_PATH,
                      monitor='mean_absolute_percentage_error',
                      save_best_only=True,
                      mode='min',
                      verbose=1)

hist = model.fit(X_train, y_train, 
                 epochs=300,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback,mdc])

#%% Model Evaluation

print(hist.history.keys())

me=ModelEvaluation()
me.Plot_Hist(hist,1,4) #to look for MAPE & val_MAPE

predicted_cases=model.predict(X_test)

graph=PlotFig()
graph.plot_fig(y_test,predicted_cases)

actual_cases=mms.inverse_transform(y_test)
predicted_inv_cases=mms.inverse_transform(predicted_cases)

graph.plot_fig(actual_cases,predicted_inv_cases)

#%% Model Analysis

print('MAE is {}'.format(mean_absolute_error(y_test,predicted_cases)))
print('MSE is {}'.format(mean_squared_error(y_test,predicted_cases)))
print('MAPE is {:.2f}%'.format(float(mean_absolute_percentage_error(y_test,predicted_cases)*100)))