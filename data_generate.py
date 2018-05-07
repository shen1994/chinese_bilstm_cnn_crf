# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:55:07 2018

@author: shen1994
"""

import codecs

import numpy as np

from fake_keras import to_categorical

def generate_batch(batch_size=None, label_class=None):
    
    batch_count = 0
    
    X = []
    Y = []
    
    while True:
        
        data_index_padding = codecs.open("data/data_index_padding.data", "r", "utf-8")
        label_index_padding = codecs.open("data/label_index_padding.data", "r", "utf-8")
        
        data_line = data_index_padding.readline()
        label_line = label_index_padding.readline()
        
        while data_line and label_line:
        
            data_str_list = data_line.strip().split()
            label_str_list = label_line.strip().split()
            
            data_list = []
            label_list = []
            for data in data_str_list:
                data_list.append(int(data))
                
            for label in label_str_list:
                label_list.append(int(label))
    
            X.append(data_list)
            Y.append(label_list)
            
            batch_count += 1
            
            if batch_count == batch_size:
                
                batch_count = 0
                
                X_ARRAY = np.array(X)
                Y_ARRAY = np.array(Y)
                
                Y_CLASS = to_categorical(Y_ARRAY, label_class).reshape((len(Y_ARRAY),len(Y_ARRAY[0]), -1))
                
                yield(X_ARRAY, Y_CLASS)
                
                X = []
                Y = []
            
            data_line = data_index_padding.readline()
            label_line = label_index_padding.readline()
            
    data_index_padding.close()
    label_index_padding.close()        
        