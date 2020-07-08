# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:12:02 2020

@author: NieZhijie
"""
import numpy as np
import pandas as pd
import jieba
from gensim import corpora, models

def tf_idf(text, stop_words_path, topK = 100):
    with open(stop_words_path, encoding='utf8') as f:
        stop_words = [l.strip() for l in f]
        refer_words = []
        words = []
        tf_dic = {}
        data = np.zeros((text.shape[0], topK))
        #分词时过滤停用词
        for sentence in text:
            words += ([x for x in jieba.cut(sentence) if x not in stop_words])
        #统计词频，找到词频前一百的词语
        for w in words:
            tf_dic[w] = tf_dic.get(w, 0) + 1        
        rank_words = [item[0] for item in sorted(tf_dic.items(), key = lambda x: x[1], reverse=True)[:topK]]
        for sentence in text:
            refer_words.append([x for x in jieba.cut(sentence) if x in rank_words])
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
    df = pd.read_csv(data_path, header=None)
    df = df[~df[1].isnull()]
    text = df[1]
    data = tf_idf(text, stop_words_path)
    data = pd.DataFrame(data)
    output = pd.concat([data, df[0]], axis=1, join='inner')
    output.to_csv("./data/data_vector.csv")