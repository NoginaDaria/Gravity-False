import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist as ppdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import NMF, LatentDirichletAllocation
from tqdm import tqdm_notebook
from scipy.stats import rankdata
from itertools import product
import enchant
from pymystem3 import Mystem
import numpy as np
import pandas as pd
import re

import gensim

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


enchant_d = enchant.Dict("ru")
mystem = SnowballStemmer("russian")
mystem_e = SnowballStemmer("english")
russian_stopwords = stopwords.words("russian")
english_stopwords = stopwords.words("english")
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text, stemmer_I):
    text = text.lower()
    #text = ' '.join(tokenizer.tokenize(text))
    if stemmer_I:
        tokens = [mystem.stem(mystem_e.stem(token)) for token in re.sub('[^a-zа-я0-9]', ' ', text).split() 
                     if not token in russian_stopwords \
                     and not token in english_stopwords \
                     #and not token.isdigit() \
                     and token.strip() not in punctuation \
                     #and not token in ['ru', 'com'] \
                     #and len(token) > 1 \
                 ]
    else:
        tokens = [token for token in re.sub('[^a-zа-я0-9]', ' ', text).split() 
                     if not token in russian_stopwords \
                     and not token in english_stopwords \
                     and not token.isdigit() \
                     and token.strip() not in punctuation \
                     #and not token in ['ru', 'com'] \
                     #and len(token) > 1 \
                 ]
        #return ''.join(Mystem().lemmatize(' '.join(tokens))[:-1])
    return ' '.join(tokens)

def process_text(doc_to_title, stemmer_I):
    for doc in tqdm_notebook(doc_to_title, total=len(doc_to_title)):
        doc_to_title[doc] = preprocess_text(doc_to_title.get(doc), stemmer_I)
    return doc_to_title

def get_page_dict(filename, stemmer_I=True):
    doc_to_title = {}
    with open(filename) as f:
        next(f)
        for line in f:
            data = line.strip().split('\t', 1)
            doc_id = int(data[0])
            if len(data) == 1:
                title = ''
            else:
                title = data[1]
            doc_to_title[doc_id] = title
    doc_to_title = process_text(doc_to_title, stemmer_I)
    doc_to_title[0] = ''
    return doc_to_title

def cosine(group, n_features, vectors):
    X = np.empty(shape=(group.size, n_features))
    for i, title in enumerate(pairwise_distances(vectors[group], metric='cosine')):
        X[i] = sorted(title, )[1:n_features + 1]     
    return X

def cvetkov_distance(x1, x2):
    try:
        if x1.shape[0] > x2.shape[0]:
            x1, x2 = x2, x1
        poses = ppdist(x1, x2, metric='cosine').argmin(axis=1)
        return 1 - (np.dot(x1.ravel(), x2[poses].ravel()) / x1.shape[0])
    except:
        return 1

def word2vec_model(w2v_fpath):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
    w2v.init_sims(replace=True)
    return w2v
    
def word2vec_vectors(docs, w2v):
    INIT_SHAPE = 100
    docs_list = []
    for doc in docs:
        doc_list = []
        for word in doc.split():
            try:
                v = w2v.get_vector(word)
            except KeyError:
                v = np.zeros(INIT_SHAPE)
            doc_list.append(v)
        docs_list.append(np.array(doc_list))
    return np.asarray(docs_list)


def cosine_add_feat(group, n_features, vectors, w_vectors, add_data=None, n_topics=20):
    X = np.empty(shape=(group.size, n_features*3 +14 ))
    dist = pairwise_distances(vectors[group], metric='cosine')
    dist_ranks = rankdata(dist).reshape(dist.shape)
    pairs_d = np.zeros((w_vectors[group].shape[0], w_vectors[group].shape[0]))
    for i, vec_i in enumerate(w_vectors[group]):
        for j, vec_j in enumerate(w_vectors[group]):
            pairs_d[i, j] = cvetkov_distance(vec_i, vec_j)

    for i, title in enumerate(dist):
        m = sorted(title, )[1:n_features + 1] 
        m_ranks = sorted(dist_ranks[i])[1:n_features + 1] 
        w2v_experiment = sorted(pairs_d[i])[1:n_features + 1] 
        X[i] = np.concatenate((m, m_ranks, w2v_experiment,
                               np.array([np.quantile(m, i) for i in np.arange(0.1, 1+0.1, 0.1)]),
                               np.array([np.mean(sorted(title, )), 
                                         np.median(sorted(title, )), 
                                         np.std(sorted(title, )),
                                         (max(sorted(title, )) +0.0001)/ (min(sorted(title, )) + 0.0001)
                                        ])
                              ))
    
    add_data_add = add_data.Text_size_log.values[group]
    X = np.c_[X, add_data_add]
    return X
        

def tf_idf(doc_to_title, doc_to_title_nstemmed, train_test_data, w2v, train=True, test=False, n_features=20, n_topics=20):
    data = pd.read_csv(train_test_data)
    groups = data.groupby('group_id')
    vectors = TfidfVectorizer().fit_transform([doc_to_title[i] for i in range(len(doc_to_title))])
    w_vectors = word2vec_vectors([doc_to_title[i] for i in range(len(doc_to_title_nstemmed))], w2v)
    add_data = pd.read_csv('add_feat.tsv', sep='\t')
    if train:
        X = np.empty(shape=(data.shape[0], n_features*3+15), dtype=np.float)
        y = np.empty(shape=(data.shape[0], ), dtype=bool)
        group_ids = np.empty(shape=(data.shape[0], ), dtype=int)

        i = 0
        for group_id, group_indx in tqdm_notebook(groups.groups.items()):
            j = i + group_indx.size
            group = data.iloc[group_indx]
            group_ids[i:j] = group_id
            y[i:j] = group.target
            X[i:j] = cosine_add_feat(group.doc_id, n_features, vectors, w_vectors, add_data)
            i = j

        return X, y, group_ids
    else:
        X = np.empty(shape=(data.shape[0], n_features*3+15), dtype=np.float)
        pair_ids = np.empty(shape=(data.shape[0], ), dtype=int)

        i = 0
        for group_id, group_indx in tqdm_notebook(groups.groups.items()):
            j = i + group_indx.size
            group = data.iloc[group_indx]
            pair_ids[i:j] = group.pair_id
            X[i:j] = cosine_add_feat(group.doc_id, n_features, vectors, w_vectors, add_data)
            i = j

        return X, pair_ids
    
def best_th(cls, X, y, groups, k=5):
    space = np.linspace(0.1, 0.9, 25)
    result = np.zeros_like(space)
    for train, test in GroupKFold(n_splits=k).split(X, y, groups):
        cls.fit(X[train], y[train])
        try:
            predict = cls.predict_proba(X[test])[:, 1]

            for i, th in enumerate(space):
                result[i] += f1_score(y[test], predict > th)
        except:
            predict = cls.predict(X[test])
            for i, th in enumerate(space):
                result[i] += f1_score(y[test], predict > th)

    best = np.argmax(result)
    return space[best], result[best] / k
 
    
def train_classifier(clf, doc_to_title, doc_to_title_nstemmed, w2v, train_data, n_features=15, k=5, train=True, test=False, scaler_I=False):
    print('preparing tf-idf matrices....')
    X, y, group_ids = tf_idf(doc_to_title, doc_to_title_nstemmed, train_data, w2v, train, test, n_features)
    if scaler_I == 'st':
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    if scaler_I == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    print('calculating scores....')

    threshold, f1 = best_th(clf, X, y, group_ids, k)

    print('\nthreshold:', threshold, '\nf1:', f1)
    return threshold, f1
    
    
def predict_test(clf, th, doc_to_title, doc_to_title_nstemmed, w2v, train_data, test_data, n_features=15, train=False, test=True, scaler_I=False):
    print('preparing tf-idf matrices....')
    X_train, y_train, group_ids_train = tf_idf(doc_to_title, doc_to_title_nstemmed, train_data, w2v, train=True, test=False, n_features = n_features)
    if scaler_I == 'st':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
    if scaler_I == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    X_test, pair_ids = tf_idf(doc_to_title, doc_to_title_nstemmed, test_data, w2v, train=False, test=True, n_features = n_features)
    if scaler_I:
        X_test = scaler.transform(X_test)
    predict = clf.predict(X_test) > th
    return predict

