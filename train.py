# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:21:56 2018

@author: shen1994
"""

import os
import argparse
import gensim
import pickle

from data_create import create_label_data

from data_preprocess import create_documents
from data_preprocess import create_useful_words
from data_preprocess import create_lexicon
from data_preprocess import create_embedding
from data_preprocess import create_label_index
from data_preprocess import create_index_label
from data_preprocess import create_matrix
from data_preprocess import maxlen_2d_list
from data_preprocess import padding_sentences

from data_generate import generate_batch

from bilstm_cnn_crf import bilstm_cnn_crf

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", help="corpus path", default="corpus", type=str)
parser.add_argument("--batch_size", help="batch size", default=256, type=int)
parser.add_argument("--epochs", help="epochs", default=3, type=int)

args = parser.parse_args()
corpus_path = args.corpus_path
batch_size = args.batch_size
epochs = args.epochs

def run():
    
    if not os.path.exists("data"):
        
        os.makedirs("data")
        
    if not os.path.exists("model"):
        
        os.makedirs("model")
    
    print("step-1--->" + u"加载词向量模型" + "--->START")
    
    embedding_model = gensim.models.Word2Vec.load(r'model/model_vector_people.m')
    
    word_dict = create_useful_words(embedding_model)
    
    embedding_size = embedding_model.vector_size
    
    print("step-2--->" + u"语料格式转换,加标注生成标准文件" + "--->START")
    
    raw_train_file = [corpus_path + os.sep + main_path + os.sep + sub_path \
                      for main_path in os.listdir(corpus_path) \
                      for sub_path in os.listdir(corpus_path + os.sep + main_path)]

    create_label_data(word_dict, raw_train_file)
    
    print("step-3--->" + u"按标点符号或是空格存储文件" + "--->START")
    
    documents_length = create_documents()

    print("step-4--->" + u"对语料中的词统计排序生成索引" + "--->START")
    
    lexicon, lexicon_reverse = create_lexicon(word_dict)
   
    print("step-5--->" + u"对所有的词创建词向量" + "--->START")
    
    useful_word_length, embedding_weights = create_embedding(embedding_model, embedding_size, lexicon_reverse)
    
    print("step-6--->" + u"生成标注以及索引" + "--->START")
      
    label_2_index = create_label_index()
    
    label_2_index_length = len(label_2_index)
    
    print("step-7--->" + u"将语料中每一句和label进行索引编码" + "--->START")
    
    create_matrix(lexicon, label_2_index)
   
    print("step-8--->" + u"将语料中每一句和label以最大长度统一长度,不足补零" + "--->START")
    
    max_len = maxlen_2d_list()
    
    padding_sentences(max_len)
    
    print("step-9--->" + u"模型创建" + "--->START")
    
    model = bilstm_cnn_crf(max_len, useful_word_length + 2, label_2_index_length, embedding_size, embedding_weights)
    
    print("step-10--->" + u"模型训练" + "--->START")
    
    if batch_size > documents_length:
        
        print("ERROR--->" + u"语料数据量过少，请再添加一些")
        
        return None

    _ = model.fit_generator(generator=generate_batch(batch_size=batch_size, label_class=label_2_index_length), \
                            steps_per_epoch=int(documents_length / batch_size), \
                            epochs=epochs, verbose=1, workers=1)
   
    print("step-11--->" + u"模型和字典保存" + "--->START")
    
    model.save_weights('model/train_model.hdf5')
    
    index_2_label = create_index_label()
    
    pickle.dump([lexicon, index_2_label], open('model/lexicon.pkl','wb'))
    
    pickle.dump([max_len, embedding_size, useful_word_length + 2, label_2_index_length], open('model/model_params.pkl','wb'))
    
    print("step-12--->" + u"打印恢复模型的重要参数" + "--->START")
    
    print("sequence_max_length: " + str(max_len))
    
    print("embedding size: " + str(embedding_size))
    
    print("useful_word_length: " + str(useful_word_length + 2))
    
    print("label_2_index_length: " + str(label_2_index_length))

    print(u"训练完成" + "--->OK")
        
if __name__  == "__main__":
    run()