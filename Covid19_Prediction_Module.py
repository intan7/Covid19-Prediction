# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 13:12:41 2022

@author: intan

This is module for Model Development.
"""

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras import Sequential,Input
import matplotlib.pyplot as plt
          

class ModelDevelopment:
    def simple_MD_model(self,X_shape,nb_class,nb_node=64, dropout_rate=0.1,
                        activation='linear'):
        
        model=Sequential()
        model.add(Input(shape=(X_shape)))
        model.add(LSTM(nb_node,return_sequences=(True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(nb_node))
        model.add(Dense(nb_class,activation))
        model.summary()
        
        return model

class ModelEvaluation:
    def Plot_Hist(self,hist,mse=2,vmse=5):
        a=list(hist.history.keys())
        plt.figure()
        plt.plot(hist.history[a[mse]])
        plt.plot(hist.history[a[vmse]])
        plt.legend(['training_'+ str(a[mse]), a[vmse]])
        plt.show()
        
class PlotFig:
    def plot_fig(self,y_test,y_pred):
        plt.figure()
        plt.plot(y_test,color='red')
        plt.plot(y_pred,color='blue')
        plt.xlabel('Day')
        plt.ylabel('No of Cases')
        plt.legend(['Actual','Predicted'])
        plt.show()