# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:12:02 2020

@author: NieZhijie
"""
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim import corpora, models

def is_useful(flat):
    include_list = ["ns", "nr", "m", "t", "u", "j", "r", "d"]
    if flat in include_list:
        return False
    return True

def get_near_words(rank_words, near_data_path):
    near_words_dic = {}
    wv_from_text = models.KeyedVectors.load_word2vec_format(near_data_path)
    near_words_vector = wv_from_text.wv
    for w in rank_words:
        try:
            near_word_list =  near_words_vector.most_similar(w)
            for item in near_word_list:
                #在近义词库中收集每个入选词的相似度大于0.7的近义词
                if item[1] >= 0.7 and item[0] not in near_words_dic.keys():
                    near_words_dic[item[0]]= w
        except KeyError:
            pass
    return near_words_dic

def tf_idf(text, stop_words_path, near_word_path, topK = 128):
    with open(stop_words_path, encoding='utf8') as f:
        stop_words = [l.strip() for l in f]
        refer_words = []
        words = []
        tf_dic = {}
        data = np.zeros((text.shape[0], topK))
        #print(data.shape)
        #分词时过滤停用词
        for sentence in text:
            words += ([x for x, flat in pseg.cut(sentence) if x not in stop_words and is_useful(flat)])
        #统计词频，找到词频前TopK的词语
        for w in words:
            tf_dic[w] = tf_dic.get(w, 0) + 1
        #寻找近义词，并一同纳入词典中
        rank_words = [item[0] for item in sorted(tf_dic.items(), key = lambda x: x[1], reverse=True)[:topK]]
        more_words = get_near_words(rank_words, near_word_path)
        refer_words = []
        for sentence in text:
            sentence_words = []
            for word in jieba.cut(sentence):
                #若存在入选词语，直接加入
                if word in rank_words:
                    sentence_words.append(word)
                #若存在入选词语的近义词，将其替换为入选词语
                elif word in more_words.keys():
                    sentence_words.append(more_words[word])
            refer_words.append(sentence_words)
        # 建立语料库词袋模型
        dictionary = corpora.Dictionary(refer_words)
        doc_vectors = [dictionary.doc2bow(word) for word in refer_words]
        # 建立语料库 TF-IDF 模型
        tf_idf = models.TfidfModel(doc_vectors)
        tf_idf_vectors = tf_idf[doc_vectors]
        for i in range(len(tf_idf_vectors)):
            for item in tf_idf_vectors[i]:
                data[i][item[0]] = item[1]
        return data
            
        
if __name__ == '__main__':
    data_path = "./data/data.csv"
    stop_words_path = "./data/stop_words.utf8"
    near_word_path = "./data/1000000-small.txt"
    df = pd.read_csv(data_path, header=None)
    df = df[~df[1].isnull()]
    text = df[1]
    data = tf_idf(text, stop_words_path, near_word_path)
    data = pd.DataFrame(data)
    output = pd.concat([data, df[0]], axis=1, join='inner')
    output.to_csv("./data/data_vector.csv")