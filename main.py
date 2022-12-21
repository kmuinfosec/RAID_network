from preprocess import contents2count, contents2count_v2
from utils import prototypeClustering, hierarchicalClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import wordpunct_tokenize
import numpy as np



def do(payloads,vec_size=512,win_size=4 ,th = 0.6 ,th2 = 0.6,min_count=3):
    payloads_indices = dict()
    for idx, payload in enumerate(payloads):
        if payload not in payloads_indices.keys():
            payloads_indices[payload] = []
        payloads_indices[payload].append(idx)

    unique_payloads = list(payloads_indices.keys())

    X = contents2count(unique_payloads, vec_size, win_size)
    prev_label_list = prototypeClustering(X, th)
    # print(prev_label_list)

    label_list = hierarchicalClustering(X, prev_label_list, th2)
    # print(label_list)

    real_label_list = [-1] * len(payloads)

    nxt_label = max(label_list) + 1
    for i in range(len(unique_payloads)):
        payload = unique_payloads[i]
        label = label_list[i]
        if label==-1 and len(payloads_indices[payload])< min_count:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = -1
        elif label==-1:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = nxt_label
            nxt_label += 1
        else:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = label

    # print(real_label_list)
    del label,payload

    clusters = dict()

    for label,payload in zip(label_list,X):
        if label not in clusters:
            clusters[label] = [payload]
        else:
            clusters.get(label).append(payload)
        

    return real_label_list,clusters


def do_v2(payloads,vec_size=512,win_size=4 ,th = 0.6 ,th2 = 0.6,min_count=3):
    payloads_indices = dict()
    for idx, payload in enumerate(payloads):
        if payload not in payloads_indices.keys():
            payloads_indices[payload] = []
        payloads_indices[payload].append(idx)

    unique_payloads = list(payloads_indices.keys())
    tf = TfidfVectorizer(tokenizer=wordpunct_tokenize,max_features=vec_size)
    X = np.array(tf.fit_transform(unique_payloads).todense())
    prev_label_list = prototypeClustering(X, th)
    # print(prev_label_list)

    label_list = hierarchicalClustering(X, prev_label_list, th2)
    # print(label_list)

    real_label_list = [-1] * len(payloads)

    nxt_label = max(label_list) + 1
    for i in range(len(unique_payloads)):
        payload = unique_payloads[i]
        label = label_list[i]
        if label==-1 and len(payloads_indices[payload])< min_count:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = -1
        elif label==-1:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = nxt_label
            nxt_label += 1
        else:
            for idx in payloads_indices[payload]:
                real_label_list[idx] = label

    # print(real_label_list)
    del label,payload

    clusters = dict()

    for label,payload in zip(label_list,X):
        if label not in clusters:
            clusters[label] = [payload]
        else:
            clusters.get(label).append(payload)
        

    return real_label_list,clusters,tf
