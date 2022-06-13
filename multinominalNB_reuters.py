
import math
import string
import operator
import copy
import numpy as np
from bs4 import BeautifulSoup

TOPIC_COUNT = {} # Counts topics
TOP_TEN_TOPICS = []
FILE_COUNT = 22
TOTAL_CLASS_NUM = 10

with open('stopwords.txt') as fp: # Load Stopwords
    STOPWORDS = set(fp.read().splitlines())

def get_words_from_article(article) -> list:
    """
        Extracts title and body from article
    """

    title = article.title
    body = article.body
    
    ret = []

    if title:
        title = title.text
        title = title.translate(str.maketrans('', '', string.punctuation)).lower() # Remove punctuation and lowercase
        title_words = remove_stopwords(title.split()) # Split words
        ret = ret + title_words

    if body:
        body = body.text
        body = body.translate(str.maketrans('', '', string.punctuation)).lower() # Remove punctuation and lowercase
        body_words = remove_stopwords(body.split()) # Split words
        ret = ret + body_words
    
    return ret


def remove_stopwords(words: list) -> list:
    """
        Remove stopwords from given list
    """
    ret = []
    
    for word in words:
        if word not in STOPWORDS:
            ret.append(word)

    return ret

def get_train_test_articles() -> tuple:
    """
        Returns train&test articles in a tuple
    """
    train_articles = []
    test_articles = []

    for file_i in range(FILE_COUNT):
        str_file_i = str(file_i).zfill(2)
        print('Processing File:', f'reut2-0{str_file_i}.sgm')

        with open(f'reuters21578/reut2-0{str_file_i}.sgm', encoding = 'latin-1') as fp:
            soup = BeautifulSoup(fp.read().lower(), 'html.parser')

        articles = soup('reuters')
        
        for article in articles:
            is_train = article.get('lewissplit') == 'train'
            
            if is_train:
                train_articles.append(article)
            else:
                test_articles.append(article)
    
    return train_articles, test_articles

def calculate_topic_counts(articles: list):
    """
        Calculates each topic count and stores them in TOPIC_COUNT dict
    """
    for article in articles:
        topics = article.topics

        if topics:
            global TOPIC_COUNT
            for elem in topics.contents:
                topic = elem.text.lower()
                if topic in TOPIC_COUNT.keys():
                    TOPIC_COUNT[topic] += 1
                else:
                    TOPIC_COUNT[topic] = 1

def find_top_ten_topics():
    """
        Find top ten topics
    """
    sorted_topic_dict = dict(
        sorted(TOPIC_COUNT.items(),
        key=operator.itemgetter(1),
        reverse=True)
    ) # Sort TOPIC_COUNT Dict by value descending

    global TOP_TEN_TOPICS
    for key, value in sorted_topic_dict.items():
        TOP_TEN_TOPICS.append(key) # Add topics

        if len(TOP_TEN_TOPICS) == TOTAL_CLASS_NUM:
            break
    
    print('Top 10 Topics:', TOP_TEN_TOPICS)

def remove_other_class_articles(articles: list) -> list:
    """
        Removes articles which are do not contain top ten topics from article list
    """
    ret = []

    for article in articles:
        topics = article.topics

        if topics:
            for elem in topics.contents:
                topic = elem.text.lower()
                if topic in TOP_TEN_TOPICS:
                    ret.append(article)
                    break

    return ret


def dot_product(a: list, b: list):
    """
        Returns dot product of given vectors(lists)
    """
    assert len(a) == len(b)
    ret = 0
    length = len(a)
    for i in range(length):
        ret += a[i] * b[i]

    return ret

def normalize_vector(a: list) -> list:
    """
        Returns normalized vector(list) of given vector(list)
    """
    ret = []
    length = 0

    for elem in range(len(a)):
        length = elem * elem

    length = math.sqrt(length)
    for i in range(len(a)):
        ret.append(a[i] / length)

    return ret

def get_topic_counts(article) -> list:
    """
        Get topic counts from article
    """
    ret = [0] * TOTAL_CLASS_NUM
    topics = article.topics
    if topics:
        for elem in topics.contents:
            topic = elem.text.lower()
            try:
                index = TOP_TEN_TOPICS.index(topic)
                ret[index] += 1
            except:
                pass

    return ret

def add_two_lists(a: list, b: list) -> list:
    """
        Returns an array which is summation of elements of given two list 
    """
    assert len(a) == len(b)
    ret = []
    for i in range(len(a)):
        ret.append(a[i] + b[i])
    return ret

class MultinomialNB:
    def __init__(self, articles):
        """
            Initialize class
        """
        self.articles = articles
        self.total_word_count_per_class = [0] * TOTAL_CLASS_NUM

    def process_data(self):
        """
            Calculates vocabulary and word count for each class
        """
        vocab = set() # vocabulary
        word_as_string_count_per_class = []
        for i in range(TOTAL_CLASS_NUM):
            word_as_string_count_per_class.append({})

        for article in self.articles:
            # for each article iterate over words and increase counts for its classes
            title = article.title
            body = article.body
            topics = article.topics
            
            if title:
                title = title.text
                title = title.translate(str.maketrans('', '', string.punctuation)).lower()
                title = title.split()
                title_words = remove_stopwords(title)

                for elem in topics.contents:
                    topic = elem.text.lower()
                    try:
                        index = TOP_TEN_TOPICS.index(topic)
                        self.total_word_count_per_class[index] += len(title_words)
                        for word in title_words:
                            if word not in word_as_string_count_per_class[index].keys():
                                word_as_string_count_per_class[index][word] = 1
                            else:
                                word_as_string_count_per_class[index][word] = word_as_string_count_per_class[index][word] + 1
                    except:
                        pass
                
                for word in title_words:
                    vocab.add(word)
                    
            if body:
                body = body.text
                body = body.translate(str.maketrans('', '', string.punctuation)).lower()
                body_words = remove_stopwords(body.split())

                for elem in topics.contents:
                    topic = elem.text.lower()
                    try:
                        index = TOP_TEN_TOPICS.index(topic)
                        self.total_word_count_per_class[index] += len(body_words)
                        for word in body_words:
                            if word not in word_as_string_count_per_class[index].keys():
                                word_as_string_count_per_class[index][word] = 1
                            else:
                                word_as_string_count_per_class[index][word] = word_as_string_count_per_class[index][word] + 1
                    except:
                        pass

                for word in body_words:
                    vocab.add(word)
        
        self.vocab_list = list(vocab)
        self.vocab_size = len(self.vocab_list)
        self.word_index = {}

        print("Vocabulary Size:", self.vocab_size)

        # give id for each word
        for i in range(len(self.vocab_list)):
            word = self.vocab_list[i]
            self.word_index[word] = i
        
        self.word_count_per_class = []

        # change word count dict to word count list by using word ids
        for i in range(TOTAL_CLASS_NUM):
            class_dict = word_as_string_count_per_class[i]
            self.word_count_per_class.append([0] * self.vocab_size)
            for key, value in class_dict.items():
                self.word_count_per_class[i][self.word_index[key]] += value

    def get_class_counts(self):
        """
            Return count of each class
        """
        counts = [0] * TOTAL_CLASS_NUM
        
        for article in self.articles:
            topics = article.topics
            for elem in topics.contents:
                topic = elem.text.lower()
                try:
                    index = TOP_TEN_TOPICS.index(topic)
                    counts[index] += 1
                except:
                    pass

        return counts

    def get_prob_for_each_class(self, class_counts):
        """
            Return probability of each class
        """
        total_class_count = sum(class_counts)
        class_probs = [0] * TOTAL_CLASS_NUM
        
        for i in range(TOTAL_CLASS_NUM):
            class_probs[i] = math.log(class_counts[i] / float(total_class_count))
        
        return class_probs

    def calculate_prob_word_per_class(self):
        """
            Calculates conditional probabilties P(word|class), uses add-one smoothing
        """
        self.prob_word_per_class = []

        for i in range(TOTAL_CLASS_NUM): # class idx
            word_counts = self.word_count_per_class[i]
            self.prob_word_per_class.append([0] * self.vocab_size)
            for j in range(len(word_counts)): # word idx
                self.prob_word_per_class[i][j] += math.log((word_counts[j] + 1.) / (self.total_word_count_per_class[i] + self.vocab_size))

        return self.prob_word_per_class

    def train_data(self):
        """
            Main function to call helper functions
        """
        self.process_data()
        class_counts = self.get_class_counts()
        class_probs = self.get_prob_for_each_class(class_counts)
        conditional_probs = self.calculate_prob_word_per_class()

        return self.word_index, class_probs, conditional_probs


def apply_mnb(train_articles, test_articles):
    """
        Applies Multinomial NB
    """
    mnb = MultinomialNB(train_articles)
    word_index, class_probs, conditional_probs = mnb.train_data() # Create Model
    
    tp = [0] * TOTAL_CLASS_NUM
    tn = [0] * TOTAL_CLASS_NUM
    fp = [0] * TOTAL_CLASS_NUM
    fn = [0] * TOTAL_CLASS_NUM

    for article in test_articles:
        scores = copy.deepcopy(class_probs) # Initial probabilities

        words = get_words_from_article(article)

        for word in words:
            for i in range(TOTAL_CLASS_NUM):
                scores[i] += conditional_probs[i][word_index.get(word, 0)] # Update class probabilities by 

        best = scores.index(max(scores)) # Get most probable class index
        topics = article.topics
        topic_indexes = []

        for elem in topics.contents:
            # retrieve topic indexes for article
            topic = elem.text.lower()
            try:
                index = TOP_TEN_TOPICS.index(topic)
                topic_indexes.append(index)
            except:
                pass
        
        # Update TruePositive, FalsePositive, TrueNegative, FalseNegative
        for i in range(TOTAL_CLASS_NUM):
            if best == i and i in topic_indexes:
                tp[i] += 1
            elif best == i and i not in topic_indexes:
                fp[i] += 1
            elif best != i and i not in topic_indexes:
                tn[i] += 1
            elif best != i and i in topic_indexes:
                fn[i] += 1
            
    # Calculate FSCORES
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

    print("MNB Result")
    print("PRECISION MICRO:", precision_micro_avg)
    print("PRECISION MACRO:", precision_macro_avg)
    print("RECALL MICRO:", recall_micro_avg)
    print("RECALL MACRO:", recall_macro_avg)
    
    micro_fscore_avg = 2 * (precision_micro_avg * recall_micro_avg) / (precision_micro_avg + recall_micro_avg)
    macro_fscore_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

    print("MICRO FSCORE:", micro_fscore_avg)
    print("MACRO FSCORE:", macro_fscore_avg)

    del mnb

class KNN:
    def __init__(self, articles):
        self.articles = articles
    
    def process_data(self):
        """
            Calculates tf-idf vector for each article
        """
        vocab = set() # vocabulary
        word_count_in_doc = [] # to use bag of words model
  
        index = 0
        for article in self.articles:
            word_count_in_doc.append({})
            words = get_words_from_article(article)
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
        
        # give an id for each word
        for i in range(self.vocab_size):
            word = vocab_list[i]
            self.word_index[word] = i
        
        self.document_freq = [0] * self.vocab_size
        term_freq = []
        for i in range(len(self.articles)):
            class_dict = word_count_in_doc[i]
            term_freq.append([0] * self.vocab_size)
            for key, value in class_dict.items():
                term_freq[i][self.word_index[key]] += value
            
            for j in range(self.vocab_size):
                if term_freq[i][j] > 0:
                    self.document_freq[j] += 1 # Update document frequency of word
                    term_freq[i][j] = math.log(1 + term_freq[i][j], 10) # calculate tf

        for i in range(self.vocab_size):
            self.document_freq[i] = math.log(len(self.articles) / float(self.document_freq[i]), 10) # calculate idf
        
        self.tf_idf = []
        for i in range(len(self.articles)): # calculates tf_idf
            self.tf_idf.append([0] * self.vocab_size)
            length = 0
            for j in range(self.vocab_size):
                self.tf_idf[i][j] = (1 + term_freq[i][j]) * (self.document_freq[j])
                length += self.tf_idf[i][j] * self.tf_idf[i][j]
            length = math.sqrt(length) # length of vector
            for j in range(self.vocab_size): # normalizes vector
                self.tf_idf[i][j] = self.tf_idf[i][j] / length

    def train_data(self):
        """
            Main function for class
        """
        self.process_data()
        
        return self.vocab_size, self.word_index, self.tf_idf, self.document_freq

def apply_knn(train_articles, test_articles, k_list):
    """
        Applies knn for each k in k_list
    """
    k_count = len(k_list)
    train_articles = train_articles[:1200]
    knn = KNN(train_articles)
    test_articles = test_articles[:800]
    vocab_size, word_index, tf_idf, document_freq = knn.train_data()
    document_count_train = len(train_articles)
    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(k_count):
        tp.append([0] * TOTAL_CLASS_NUM)
        tn.append([0] * TOTAL_CLASS_NUM)
        fp.append([0] * TOTAL_CLASS_NUM)
        fn.append([0] * TOTAL_CLASS_NUM)

    for article in test_articles:
        scores = {}
        vector = [0] * vocab_size 

        words = get_words_from_article(article)
        
        for word in words:
            index = word_index.get(word)
            if index:
                vector[index] += 1
                
        for i in range(vocab_size):
            if vector[i]:
                vector[i] = math.log(1 + vector[i], 10) * math.log(document_count_train / float(document_freq[i]), 10)

        vector = normalize_vector(vector) # normalize tf-idf vector

        for i in range(document_count_train):
            scores[i] = np.dot(tf_idf[i], vector) # dot product for cosine similarity
        
        sorted_scores = dict(
            sorted(scores.items(),
            key=operator.itemgetter(1),
            reverse=True)
        ) # sort distances

        sum_topic_count = [0] * TOTAL_CLASS_NUM

        count = 0
        pointer = 0
        
        topic_indexes = [] 
        topics = article.topics

        for elem in topics.contents: # Get topic indexes of the article
            topic = elem.text.lower()
            try:
                index = TOP_TEN_TOPICS.index(topic)
                topic_indexes.append(index)
            except:
                pass
        
        # Iterate over nearest neighbours
        for key, value in sorted_scores.items():
            if count == k_list[-1]+1:
                break
            
            if k_list[pointer] == count: # If k neighbours added
                max_topic_count = max(sum_topic_count)
                my_topics = []
                for i in range(TOTAL_CLASS_NUM):
                    if sum_topic_count[i] == max_topic_count:
                        my_topics.append(i)
                
                # Update TruePositive, FalsePositive, TrueNegative, FalseNegative
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

            topic_counts = get_topic_counts(train_articles[key])
            sum_topic_count = add_two_lists(topic_counts, sum_topic_count)

            count += 1

    best_k = 0
    best_avg = -1
    print("KNN RESULTS")
    for j in range(k_count): # Calculate FSCORES for each k
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

        print("PRECISION MICRO:", precision_micro_avg)
        print("PRECISION MACRO:", precision_macro_avg)
        print("RECALL MICRO:", recall_micro_avg)
        print("RECALL MACRO:", recall_macro_avg)

        micro_fscore_avg = 2 * (precision_micro_avg * recall_micro_avg) / (precision_micro_avg + recall_micro_avg)
        macro_fscore_avg = 2 * (precision_macro_avg * recall_macro_avg) / (precision_macro_avg + recall_macro_avg)

        print("MICRO FSCORE:", micro_fscore_avg)
        print("MACRO FSCORE:", macro_fscore_avg)

        fscore_avg = (micro_fscore_avg + macro_fscore_avg) / 2

        print("FSCORE AVG:", fscore_avg)

        if fscore_avg > best_avg:
            best_avg = fscore_avg
            best_k = k_list[j]
    
    return best_k
    
def preprocess():
    train_articles, test_articles = get_train_test_articles()
    
    print('Total Train Article Count:', len(train_articles))
    print('Total Test Article Count:', len(test_articles))

    calculate_topic_counts(train_articles)
    find_top_ten_topics()

    train_articles = remove_other_class_articles(train_articles)
    test_articles = remove_other_class_articles(test_articles)

    print('Train Article Count After Removing According to Top Topics:', len(train_articles))
    print('Test Article Count After Removing According to Top Topics:', len(test_articles))
    
    global TOPIC_COUNT
    del TOPIC_COUNT # Top 10 topics are calculated, Release memory

    return train_articles, test_articles


def main(args=None):
    train_articles, test_articles = preprocess()

    apply_mnb(train_articles, test_articles)
    
    """
        Tuning Part
        Commented out due to performance issues
        part_len = len(train_articles) // 10
        count_best_k = {1:0, 3:0, 5:0, 7:0, 9:0}
        overall_best_k = 1
        overall_best_count = 0
        for part in range(10):
            start_index = part_len * part
            end_index = part_len * part + part_len
            dev_articles = train_articles[start_index:end_index]
            if part == 0:
                train_temp_articles = train_articles[end_index:]
            else
                train_temp_articles = train_articles[:start_index] + train_articles[end_index:]
            
            best_k = apply_knn(train_temp_articles, dev_articles, [1,3,5,7,9])
            count_best_k[best_k] = count_best_k[best_k] + 1
            if count_best_k[best_k] > overall_best_count:
                overall_best_count = count_best_k[best_k]
                overall_best_k = best_k
        
        apply_knn(train_articles, test_articles, [best_k])
    """
    
    apply_knn(train_articles, test_articles, [3])


if __name__ == '__main__':
    main()
