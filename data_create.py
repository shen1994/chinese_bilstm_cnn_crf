# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:38:29 2018

@author: shen1994
"""

import re
import codecs

def number_to_character(word):
    """数字转换为中文,但是未考虑实际大小"""
    if word == "0" or word == 0:
        return u"零"
    elif word == "1" or word == 1:
        return u"一"
    elif word == "2":
        return u"二"
    elif word == "3":
        return u"三"
    elif word == "4":
        return u"四"
    elif word == "5":
        return u"五"
    elif word == "6":
        return u"六"
    elif word == "7":
        return u"七"
    elif word == "8":
        return u"八"
    elif word == "9":
        return u"九"
    else:
        return word
        
def special_charater_filter(word):
    """过滤形如  的 后面的承接字 """
    new_word = ""
    word_match = re.match("\u7684/ude(\d+)", word)
    if word_match:
        noice_word = word_match.group(0).replace(u"的", "")
        new_word = word.replace(noice_word, "")
        return new_word
        
    return word

  
def single_word_filter(word_list):
    
    new_word_list = []
    
    for w in word_list:
        
        new_word = ""
        w = special_charater_filter(w)

        for sub_w in w:
            sub_w = number_to_character(sub_w)
            new_sub_word = re.match("[\u4e00-\u9fa5，；。！？]", sub_w)
            if new_sub_word:
                new_sub_word = new_sub_word.group(0)
                new_word += new_sub_word
                
        if new_word:
            new_word_list.append(new_word)
            
    return new_word_list


def create_label_data(word_dict, file_list):
    
    file_des = codecs.open("data/train.data", "w", "utf-8")

    for file_src in file_list:
        with codecs.open(file_src, "r", "utf-8") as infile:
            lines = infile.readlines()
            for line in lines:
                new_word_list = []
                word_list = line.strip().split()

                new_word_list = single_word_filter(word_list)

                for word in new_word_list:
                    useful_word_flag = True
                    for w in word:
                        if w not in word_dict:
                            useful_word_flag = False
                            break
                    if useful_word_flag:
                        if len(word) == 1:
                            file_des.write(word + "\tS\n")
                        else:
                            file_des.write(word[0] + "\tB\n")
                            for w in word[1:-1]:
                                file_des.write(w + "\tM\n")
                            file_des.write(word[-1] + "\tE\n")
                    else:
                        for w in word:
                            file_des.write(w + "\tU\n")
                file_des.write("\n")

        print(u"加载数据" + "--->" + str(file_src) + "--->OK")
                
    file_des.close()
    