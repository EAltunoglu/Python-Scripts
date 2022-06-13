from ast import arg
import math
import string
import operator
import copy
import csv
import json
import threading
from multiprocessing import Pool
import numpy as np
import pandas as pd

from FileReader import *

"""
Notlar:
title,abstract, label string listesi

"""

VOCABULARY = set()
IS_LABEL_INT = False

CLASSES = [
    'treatment',  # 0
    'diagnosis',  # 1
    'prevention',  # 2
    'mechanism',  # 3
    'transmission',  # 4
    'epidemic forecasting',  # 5
    'case report'  # 6
]
TOTAL_CLASS_NUM = 7

WORD_INDEX = {}
PROCESS_INDEXES = [
    2, # TITLE
    3, # ABSTRACT
]
PROCESS_FIELDS = set(['title', 'abstract'])

lock = 0


class MultinomialNB:
    def __init__(self, documents):
        self.documents = documents
        self.total_word_count_per_class = [0] * TOTAL_CLASS_NUM

    def process_data(self):
        vocab = set()
        word_as_string_count_per_class = []

        for i in range(TOTAL_CLASS_NUM):
            word_as_string_count_per_class.append({})

        for document in self.documents:
            words = document.title + document.abstract
            for _class in document.label:
                index = CLASSES.index(_class)
                self.total_word_count_per_class[index] += len(words)

                for word in words:
                    if word not in word_as_string_count_per_class[index].keys():
                        word_as_string_count_per_class[index][word] = 1
                    else:
                        word_as_string_count_per_class[index][word] = word_as_string_count_per_class[index][word] + 1

                for word in words:
                    vocab.add(word)

        self.vocab_list = list(vocab)
        self.vocab_size = len(self.vocab_list)
        self.word_index = {}

        for i in range(len(self.vocab_list)):
            word = self.vocab_list[i]
            self.word_index[word] = i

        self.word_count_per_class = []

        for i in range(TOTAL_CLASS_NUM):
            class_dict = word_as_string_count_per_class[i]
            self.word_count_per_class.append([0] * self.vocab_size)

            for key, value in class_dict.items():
                self.word_count_per_class[i][self.word_index[key]] += value

    def get_class_counts(self):
        counts = [0] * TOTAL_CLASS_NUM

        for document in self.documents:
            for _class in document.label:
                index = CLASSES.index(_class)
                counts[index] += 1

        return counts

    def get_prob_for_each_class(self, class_counts):
        total_class_count = sum(class_counts)
        class_probs = [0] * TOTAL_CLASS_NUM

        for i in range(TOTAL_CLASS_NUM):
            class_probs[i] = math.log(class_counts[i] / float(total_class_count))

        return class_probs

    def calculate_prob_word_per_class(self):
        self.prob_word_per_class = []

        for i in range(TOTAL_CLASS_NUM):
            word_counts = self.word_count_per_class[i]
            self.prob_word_per_class.append([0] * self.vocab_size)
            for j in range(len(word_counts)):
                self.prob_word_per_class[i][j] += math.log(
                    ((word_counts[j] + 1.) / (self.total_word_count_per_class[i] + self.vocab_size)))

                # self.prob_word_per_class[i][j] += math.log(((word_counts[j] + 1.) / (self.total_word_count_per_class[i] + self.vocab_size))*self.vocab_size/10)

        return self.prob_word_per_class

    def train_data(self):
        self.process_data()
        class_counts = self.get_class_counts()
        class_probs = self.get_prob_for_each_class(class_counts)
        conditional_probs = self.calculate_prob_word_per_class()

        return self.word_index, class_probs, conditional_probs


with open('stopwords.txt') as fp: # Load Stopwords
    STOPWORDS = set(fp.read().splitlines())


def decode_label(val: int) -> list:
    ans = []

    for i in range(len(CLASSES)):
        if ((val >> i) & 1) == 1:
            ans.append(CLASSES[i])

    return ans


def dot_product(a: dict, b: dict) -> float:
    ret = 0

    if len(a) < len(b):
        for key, value in a.items():
            ret += value * b.get(key, 0)
    else:
        for key, value in b.items():
            ret += value * a.get(key, 0)
    
    # assert ret >= 0
    # assert ret <= 1

    return ret


def get_class_counts(document: Document) -> list:
    ret = [0] * TOTAL_CLASS_NUM
    
    for _class in document.label:
        index = CLASSES.index(_class)
        ret[index] += 1
    
    return ret


def normalize_vector(a: dict) -> dict:
    ret = {}
    length = 0

    for key, value in a.items():
        length += value * value

    length = math.sqrt(length)
    for key, value in a.items():
        ret[key] = value / length
    
    # assert abs(get_vector_length(ret) - 1) < 0.001

    return ret


def get_vector_length(a: dict) -> float:
    length = 0
    
    for key, value in a.items():
        length += value * value
    
    length = math.sqrt(length)

    return length


def add_two_lists(a: list, b: list) -> list:
    assert len(a) == len(b)
    ret = []
    
    for i in range(len(a)):
        ret.append(a[i] + b[i])
    
    return ret


def expo(array,base):
    return np.exp(np.log(10)*np.array(array)).tolist()





def apply_mnb(train_data, test_data, label_int: bool):
    mnb = MultinomialNB(train_data)
    word_index, class_probs, conditional_probs = mnb.train_data()
    
    tp = [0.00000007] * TOTAL_CLASS_NUM
    tn = [0.00000007] * TOTAL_CLASS_NUM
    fp = [0.00000007] * TOTAL_CLASS_NUM
    fn = [0.00000007] * TOTAL_CLASS_NUM

    goldStandart=[]
    predictions=[]

    for idx, document in enumerate(test_data):
        scores = copy.deepcopy(class_probs)

        words = document.title + document.abstract

        for word in words:
            for i in range(TOTAL_CLASS_NUM):
                scores[i] += conditional_probs[i][word_index.get(word, 0)]

        if label_int:
            raise Exception("NOT IMPLEMENTED")

        best = scores.index(max(scores)) 
        #scores=expo(scores,10)

        topic_indexes = []
        
        for _class in document.label:
            index = CLASSES.index(_class)
            topic_indexes.append(index)

        topics_dump=[idx]
        
        scores=expo(scores,10)
        total=np.array(sum(scores))
        scores=np.array(scores)
        #scores=scores/total 
        scores=scores.tolist()
        scores.insert(0,idx)
        #print(scores)
        predictions.append(scores)

        for i in range(TOTAL_CLASS_NUM):
            if i in topic_indexes:
                topics_dump.append(1)
            else:
                topics_dump.append(0)
        goldStandart.append(topics_dump)
        
        for i in range(TOTAL_CLASS_NUM):
            if best == i and i in topic_indexes:
                tp[i] += 1
            elif best == i and i not in topic_indexes:
                fp[i] += 1
            elif best != i and i not in topic_indexes:
                tn[i] += 1
            elif best != i and i in topic_indexes:
                fn[i] += 1
            
    tp_total = sum(tp)
    fp_total = sum(fp)
    fn_total = sum(fn)
    precision_micro_avg = tp_total / (tp_total + fp_total)
    precision_macro_avg = 0
    for i in range(TOTAL_CLASS_NUM):
        precision_macro_avg += tp[i]/(tp[i] + fp[i])
    precision_macro_avg /= TOTAL_CLASS_NUM

    recall_micro_avg = tp_total / (tp_total + fn_total)
    recall_macro_avg = 0
    for i in range(TOTAL_CLASS_NUM):
        recall_macro_avg += tp[i] / (tp[i] + fn[i])
    recall_macro_avg /= TOTAL_CLASS_NUM

    df = pd.DataFrame(goldStandart)
    df.columns =['0', '1', '2', '3','4', '5', '6', '7']
    df.rename({'0': 'PMID', '1': 'Treatment','2': 'Diagnosis','3': 'Prevention','4': 'Mechanism','5': 'Transmission','6': 'Epidemic Forecasting','7': 'Case Report'},  axis=1, inplace=True)
    
    df.to_csv('goldStandart.csv', index=False)

    df = pd.DataFrame(predictions)
    #print(predictions)
    df.columns =['0', '1', '2', '3','4', '5', '6', '7']
    df.rename({'0': 'PMID', '1': 'Treatment','2': 'Diagnosis','3': 'Prevention','4': 'Mechanism','5': 'Transmission','6': 'Epidemic Forecasting','7': 'Case Report'},  axis=1, inplace=True)
    
    df.to_csv('predictions.csv', index=False)
    #print(df.columns)
    #print(len(test_data))
    print("MNB Result")
    print("PRECISION MICRO:", precision_micro_avg)
    print("PRECISION MACRO:", precision_macro_avg)
    print("RECALL MICRO:", recall_micro_avg)
    print("RECALL MACRO:", recall_macro_avg)
    
    micro_fscore_avg = 2 * (precision_micro_avg * recall_micro_avg) / (precision_micro_avg + recall_micro_avg)
    macro_fscore_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

    print("MICRO FSCORE:", micro_fscore_avg)
    print("MACRO FSCORE:", macro_fscore_avg)

    print("FSCORE AVERAGE:", (micro_fscore_avg + macro_fscore_avg) / 2.)

    print("ClassName ---  TruePositive ---  TrueNegative ---  FalsePositive ---  FalseNegative")
    for i in range(TOTAL_CLASS_NUM):
        print(CLASSES[i],"          ",tp[i],"           ", tn[i],"        ", fp[i],"             ", fn[i] )

    del mnb


class KNN:
    def __init__(self, documents):
        self.documents = documents
    
    def process_data(self):
        vocab = set()
        word_count_in_doc = []
  
        index = 0
        for document in self.documents:
            word_count_in_doc.append({})
            words = document.title + document.abstract
            for word in words:
                vocab.add(word)
                if word not in word_count_in_doc[index].keys():
                    word_count_in_doc[index][word] = 1
                else:
                    word_count_in_doc[index][word] = word_count_in_doc[index][word] + 1
            index += 1
        
        vocab_list = list(vocab)
        self.vocab_size = len(vocab_list)
        self.word_index = {}
        print("VOCAB_SIZE: ",self.vocab_size)

        for i in range(self.vocab_size):
            word = vocab_list[i]
            self.word_index[word] = i
        
        self.document_freq = [0] * self.vocab_size
        term_freq = []
        
        for i in range(len(self.documents)):
            class_dict = word_count_in_doc[i]
            term_freq.append({})
            
            for key, value in class_dict.items():
                if self.word_index[key] in term_freq[i].keys():
                    term_freq[i][self.word_index[key]] += value
                else:
                    term_freq[i][self.word_index[key]] = value

            for key, value in term_freq[i].items():
                self.document_freq[key] += 1
                term_freq[i][key] = math.log(1 + value, 10)

        for i in range(self.vocab_size):
            self.document_freq[i] = math.log(len(self.documents) / float(self.document_freq[i]), 10)
        
        self.tf_idf = []
        for i in range(len(self.documents)):
            self.tf_idf.append({})
            length = 0

            for key, value in term_freq[i].items():
                ans = (1 + value) * (self.document_freq[key])
                self.tf_idf[i][key] = ans
                length += ans * ans

            length = math.sqrt(length)
            
            for key, value in self.tf_idf[i].items():
                self.tf_idf[i][key] = value / length
            

    def train_data(self):
        self.process_data()
        
        return self.word_index, self.tf_idf, self.document_freq

def apply_knn(train_data, test_data, k_list, is_tuning: bool, label_int: bool = False):
    k_count = len(k_list)
    knn = KNN(train_data)
    word_index, tf_idf, document_freq = knn.train_data()
    document_count_train = len(train_data)
    test_data= test_data[:20]
    tp = []
    tn = []
    fp = []
    fn = []

    if label_int:
        raise Exception("NOT IMPLEMENTED")

    for i in range(k_count):
        tp.append([0.0000000007] * TOTAL_CLASS_NUM)
        tn.append([0.0000000007] * TOTAL_CLASS_NUM)
        fp.append([0.0000000007] * TOTAL_CLASS_NUM)
        fn.append([0.0000000007] * TOTAL_CLASS_NUM)

    for idx,document in enumerate(test_data):
        scores = {}
        vector = {}

        words = document.title + document.abstract
        
        for word in words:
            index = word_index.get(word)
            if index:
                if index in vector.keys():
                    vector[index] += 1
                else:
                    vector[index] = 1
        
        for key, value in vector.items():
            vector[key] = math.log(1 + value, 10) * math.log(document_count_train / float(document_freq[key]), 10)

        vector = normalize_vector(vector)

        process_list = []

        for i in range(document_count_train):
            process_list.append((tf_idf[i], vector))

        global lock
        lock = 0

        def calculate_dot_product(begin, end):
            global lock
            for i in range(begin, end):
                scores[i] = dot_product(tf_idf[i],vector)
            lock = lock + 1
        
        num_of_threads = 12
        thread_pool = []
        for i in range(num_of_threads):
            begin = i * document_count_train // num_of_threads
            end = begin + document_count_train // num_of_threads-1
            if i == num_of_threads-1:
                end = document_count_train-1
            thread_pool.append(threading.Thread(target=calculate_dot_product, args=(begin, end,)))
            thread_pool[i].start()
        while lock < 12:
            pass

        sorted_scores = dict(
            sorted(scores.items(),
            key=operator.itemgetter(1),
            reverse=True)
        )

        sum_topic_count = [0] * TOTAL_CLASS_NUM

        with open('readme.txt', 'w') as f:
            f.write(str(sorted_scores))
        count = 0
        pointer = 0
        
        topic_indexes = [] 

        for _class in document.label:
            index = CLASSES.index(_class)
            topic_indexes.append(index)
        
        for key, value in sorted_scores.items():
            if count == k_list[-1]+1:
                break
            
            if k_list[pointer] == count:
                max_topic_count = max(sum_topic_count)
                my_topics = []
                for i in range(TOTAL_CLASS_NUM):
                    if sum_topic_count[i] == max_topic_count:
                        my_topics.append(i)
                
                for i in range(TOTAL_CLASS_NUM):
                    if i in my_topics and i in topic_indexes:
                        tp[pointer][i] += 1
                    elif i in my_topics and i not in topic_indexes:
                        fp[pointer][i] += 1
                    elif i not in my_topics and i not in topic_indexes:
                        tn[pointer][i] += 1
                    elif i not in my_topics and i in topic_indexes:
                        fn[pointer][i] += 1

                pointer += 1

            class_counts = get_class_counts(train_data[key])
            sum_topic_count = add_two_lists(class_counts, sum_topic_count)

            count += 1

    best_k = 0
    best_avg = -1
    if not is_tuning:
        print("KNN RESULTS")
    for j in range(k_count):
        if not is_tuning:
            print("FOR K =", k_list[j])
        tp_total = sum(tp[j])
        fp_total = sum(fp[j])
        fn_total = sum(fn[j])
        precision_micro_avg = tp_total / (tp_total + fp_total)
        precision_macro_avg = 0
        for i in range(TOTAL_CLASS_NUM):
            precision_macro_avg += tp[j][i]/(tp[j][i] + fp[j][i])
        precision_macro_avg /= TOTAL_CLASS_NUM

        recall_micro_avg = tp_total / (tp_total + fn_total)
        recall_macro_avg = 0
        for i in range(TOTAL_CLASS_NUM):
            recall_macro_avg += tp[j][i] / (tp[j][i] + fn[j][i])
        recall_macro_avg /= TOTAL_CLASS_NUM
        
        if not is_tuning:
            print("PRECISION MICRO:", precision_micro_avg)
            print("PRECISION MACRO:", precision_macro_avg)
            print("RECALL MICRO:", recall_micro_avg)
            print("RECALL MACRO:", recall_macro_avg)

        micro_fscore_avg = 2 * (precision_micro_avg * recall_micro_avg) / (precision_micro_avg + recall_micro_avg)
        macro_fscore_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

        if not is_tuning:
            print("MICRO FSCORE:", micro_fscore_avg)
            print("MACRO FSCORE:", macro_fscore_avg)

        fscore_avg = (micro_fscore_avg + macro_fscore_avg) / 2
        
        if not is_tuning:
            print("FSCORE AVG:", fscore_avg)

        if fscore_avg > best_avg:
            best_avg = fscore_avg
            best_k = k_list[j]
    
    del knn

    return best_k

def tune_k(train_data: list) -> int:
    part_len = len(train_data) // 10
    count_best_k = {1:0, 3:0, 5:0, 7:0, 9:0}
    overall_best_k = 1
    overall_best_count = 0
    
    for part in range(10):
        start_index = part_len * part
        end_index = start_index + part_len
        dev_documents = train_data[start_index:end_index]
        
        if part == 0:
            train_temp_documents = train_data[end_index:]
        else:
            train_temp_documents = train_data[:start_index] + train_data[end_index:]
        
        best_k = apply_knn(train_temp_documents, dev_documents, [1,3,5,7,9], True)
        count_best_k[best_k] = count_best_k[best_k] + 1
        
        if count_best_k[best_k] > overall_best_count:
            overall_best_count = count_best_k[best_k]
            overall_best_k = best_k
        
    return overall_best_k


"""
sadece title
KNN RESULTS
BEST K: 7
FOR K = 7
PRECISION MICRO: 0.76394491584365
PRECISION MACRO: 0.749641851924527
RECALL MICRO: 0.6456618857277217
RECALL MACRO: 0.4763261439899616
MICRO FSCORE: 0.6998407136030583
MACRO FSCORE: 0.5825176740186448
FSCORE AVG: 0.6411791938108515
"""
"""
title + abstract
VOCAB_SIZE:  84785
VOCAB_SIZE:  84780
VOCAB_SIZE:  84752
VOCAB_SIZE:  84911
VOCAB_SIZE:  85065
VOCAB_SIZE:  85042
VOCAB_SIZE:  84922
VOCAB_SIZE:  84756
VOCAB_SIZE:  84954
VOCAB_SIZE:  84901
BEST K: 9
KNN RESULTS
FOR K = 9
PRECISION MICRO: 0.8748184722625617
PRECISION MACRO: 0.8218211922507038
RECALL MICRO: 0.7082059722548789
RECALL MACRO: 0.5870566613949499
MICRO FSCORE: 0.7827442827442828
MACRO FSCORE: 0.6848792521479405
FSCORE AVG: 0.7338117674461117
"""