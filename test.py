# -*- coding: utf-8 -*-
"""
Created on Sun May  6 21:25:48 2018

@author: shen1994
"""

import pickle

import numpy as np

from fake_keras import pad_sequences
from bilstm_cnn_crf import bilstm_cnn_crf

def predict_one_text(text, label):
    
    start_index = len(label) - len(text)
    
    label = label[start_index:]

    segment_text = ""
    for p, t in zip(label, text):
        if p in [0, 3, 4, 5]:
            segment_text += (t + " ")
        else:
            segment_text += t
            
    return segment_text
    
def predict_many_text(text_list, model, lexicon, maxlen):
    
    new_text_list = []

    for text in text_list:
        temp = []
        for c in text:
            if c in lexicon:
                temp.append(lexicon.get(c))
            else:
                temp.append(-1)
        new_text_list.append(temp)
        
    test_array = pad_sequences(new_text_list, maxlen=maxlen)
    
    test_pred = model.predict(test_array, verbose=1)
    
    label_list = np.argmax(test_pred,axis=2)
    
    new_text_list = []
    for text, label in zip(text_list, label_list):
        new_text = predict_one_text(text, label)
        new_text_list.append(new_text)
        
    return new_text_list

def run():
    
    sequence_max_length, embedding_size, \
    useful_word_length, label_2_index_length = pickle.load(open('model/model_params.pkl','rb'))
    
    model = bilstm_cnn_crf(sequence_max_length, useful_word_length,\
                           label_2_index_length, embedding_size, is_train=False)
    
    model.load_weights('model/train_model.hdf5')
    
    lexicon, index_2_label = pickle.load(open('model/lexicon.pkl','rb'))
    
    text_list = [u"投资美国股市的明智做法是追着傻钱跑", u"其实就是买入并持有美国股票这样的普通组合"]
    
    new_text_list = predict_many_text(text_list, model, lexicon, sequence_max_length)
    
    print(new_text_list)

if __name__ == "__main__":
    run()
    