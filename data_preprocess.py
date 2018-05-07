# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:12:11 2018

@author: shen1994
"""

import codecs
import numpy as np

def create_documents():
    """ 按标点符号或是空格存储文件 """
    documents_length = 0
    chars,labels = [],[]

    chars_file = codecs.open("data/data.data", 'w', 'utf-8')
    labels_file = codecs.open("data/label.data", 'w', 'utf-8')
    

    with codecs.open("data/train.data", 'r', 'utf-8') as f:
        for line in f:

            line=line.strip()
			
            if len(line)==0:
                if len(chars)!=0:
                    for char in chars:
                        chars_file.write(char + "\t")
                    chars_file.write("\n")
                    for label in labels:
                        labels_file.write(label + "\t")
                    labels_file.write("\n")
                    documents_length += 1
                    chars, labels=[], []

            else:
                pieces=line.strip().split()
                chars.append(pieces[0])
                labels.append(pieces[1])

                if pieces[0] in ['。','，','；','！','？']:
                    
                    for char in chars:
                        chars_file.write(char + "\t")
                    chars_file.write("\n")
                    for label in labels:
                        labels_file.write(label + "\t")
                    labels_file.write("\n")
                    
                    documents_length += 1
                    chars, labels=[], []

        if len(chars)!=0:
            
            for char in chars:
                chars_file.write(char + "\t")
            chars_file.write("\n")
            for label in labels:
                labels_file.write(label + "\t")
            labels_file.write("\n")
            
            documents_length += 1
            chars, labels=[], []

    chars_file.close()
    labels_file.close()
    
    return documents_length
    
def create_useful_words(embedding_model):
    
    return list(embedding_model.wv.vocab.keys())
    
def create_lexicon(word_dict):
    """ 生成词典 """
    chars = {}
    # 统计词出现的次数
    with codecs.open("data/data.data", 'r', 'utf-8') as f:
        line = f.readline()
        while(line):
            
            book_chars = line.strip().split()
            for sequence in book_chars:
                for char in sequence:
                    chars[char] = chars.get(char,0) + 1

            line = f.readline()

    sorted_chars = sorted(chars.items(), key=lambda x:x[1], reverse=True)

    # 下标从1开始 0用来补长
    lexicon = dict([(item[0],index+1) for index, item in enumerate(sorted_chars)])
    
    del sorted_chars
    
    # 替换无用词的标记,标记为-1
    for v in lexicon:
        if v not in word_dict:
            lexicon[v] = -1

    lexicon_reverse = dict(zip(lexicon.values(), lexicon.keys()))
    
    return lexicon, lexicon_reverse
    
def create_label_index(): 

    return {'P':0, 'B':1, 'M':2, 'E':3, 'S':4, 'U':5}

def create_index_label(): 

    return {0:'Pad',1:'B',2:'M',3:'E',4:'S',5:'Unk'}
    
def create_embedding(embedding_model, embedding_size, lexicon_reverse):
    
    word_dict = create_useful_words(embedding_model)
    
    useful_word = []
    useful_word_length = 0
    for word in list(lexicon_reverse.values()):
        if word in word_dict:
            useful_word_length += 1
            useful_word.append(word)
    
    del word_dict
      
    # 增加 padding 和 unknown
    embedding_weights = np.zeros((useful_word_length + 2, embedding_size))
    
    for i in range(useful_word_length):
        embedding_weights[i + 1] = embedding_model.wv[useful_word[i]]

    # 无效词嵌入向量
    embedding_weights[-1] = np.random.uniform(-1, 1, embedding_size)
    
    return useful_word_length, embedding_weights
    
def create_matrix(lexicon, label_2_index):

    data_index = codecs.open("data/data_index.data", 'w', 'utf-8')
    label_index = codecs.open("data/label_index.data", 'w', 'utf-8')

    file_chars = codecs.open("data/data.data", 'r', 'utf-8')
    file_labels = codecs.open("data/label.data", 'r', 'utf-8')
    
    chars_line = file_chars.readline()
    labels_line = file_labels.readline()
    
    while (chars_line and labels_line):
        
        book_chars = chars_line.strip().split()
        book_labels = labels_line.strip().split()
        
        for char, label in zip(book_chars, book_labels):
            data_index.write(str(lexicon[char]) + "\t")
            label_index.write(str(label_2_index[label]) + "\t")
            
        data_index.write("\n")
        label_index.write("\n")
        
        chars_line = file_chars.readline()
        labels_line = file_labels.readline()
        
    file_chars.close()
    file_labels.close()
    
    data_index.close()
    label_index.close()
    
def padding_sentences(max_len):
    
    data_index = codecs.open("data/data_index.data", 'r', 'utf-8')
    label_index = codecs.open("data/label_index.data", 'r', 'utf-8')
    
    data_index_padding = codecs.open("data/data_index_padding.data", 'w', 'utf-8')
    label_index_padding = codecs.open("data/label_index_padding.data", 'w', 'utf-8')
    
    data_line = data_index.readline()
    
    while data_line:
        
        book_data = data_line.strip().split()
        
        book_data_len = len(book_data)
        
        new_book_data = []
        
        if book_data_len < max_len:
            new_book_data = ([str(0)] * (max_len - book_data_len) + book_data)
        else:
            new_book_data = book_data
            
        for data_word in new_book_data:
            
            data_index_padding.write(data_word + "\t")
        
        data_index_padding.write("\n")
        
        data_line = data_index.readline()

    label_line = label_index.readline()
    
    while label_line:
        
        book_label = label_line.strip().split()
        
        book_label_len = len(book_label)
        
        new_book_label = []

        if book_label_len < max_len:
            new_book_label = ([str(0)] * (max_len - book_label_len) + book_label)
        else:
            new_book_label = book_label
            
        for label_word in new_book_label:
            
            label_index_padding.write(label_word + "\t")
        
        label_index_padding.write("\n")
        
        label_line = label_index.readline()
        
    data_index.close()
    label_index.close()
    data_index_padding.close()
    label_index_padding.close()
    
def maxlen_2d_list():
    
    max_len = 0
    
    data_index = codecs.open("data/data_index.data", 'r', 'utf-8')
    
    data_line = data_index.readline()
    
    while data_line:
        
        book_data = data_line.strip().split()
        
        book_data_len = len(book_data)
        
        if book_data_len > max_len:
            
            max_len = book_data_len
        
        data_line = data_index.readline()
        
    data_index.close()
    
    return max_len    
    