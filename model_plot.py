# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:07:51 2018

@author: shen1994
"""

import pickle

from keras.utils import plot_model

from bilstm_cnn_crf import bilstm_cnn_crf

if __name__ == "__main__":
    
    sequence_max_length, embedding_size, \
    useful_word_length, label_2_index_length = pickle.load(open('model_params.pkl','rb'))
    
    model = bilstm_cnn_crf(sequence_max_length, useful_word_length,\
                           label_2_index_length, embedding_size, is_train=False)
    
    model.load_weights('train_model.hdf5')
    
    plot_model(model, to_file='bilstm_cnn_crf_model.png', show_shapes=True, show_layer_names=True)
    
    print(u"模型绘制完成" + "--------------OK")
    