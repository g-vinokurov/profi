# -*- coding: utf-8 -*-

import nltk

# Download necessary libs
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pymorphy2 import MorphAnalyzer

import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import math
import os

STOP_WORDS = set(stopwords.words('russian'))
PUNCTUATIONS = ',.;:\'\"[]()?!-'


def load_dataset(mode: str):
    if mode != 'train' and mode != 'test':
        return []

    # Load train or test dataset
    folder = os.path.join('texts', mode)

    # Read each file from folder
    # Letter in the end of file name means class of text
    texts = []
    classes = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), encoding='utf-8') as file:
            texts.append(file.read())
        classes.append(filename.split('.')[-2])
    return texts, classes


def preprocess(text: str):
    # Make lowercase
    text = text.lower()

    # Split text on tokens (punctuation marks, words)
    tokens = word_tokenize(text, language='russian')

    # Remove stopwords (words without useful meaning)
    tokens = [x for x in tokens if x not in STOP_WORDS]

    # Remove punctuation marks
    tokens = [x for x in tokens if x not in PUNCTUATIONS]

    # Convert words to normal form
    morph = MorphAnalyzer()
    lemmas = [(x, morph.normal_forms(x)) for x in tokens]
    
    # If word doesn't have a normal form, we don't touch it
    lemmas = [normals[0] if normals else x for x, normals in lemmas]

    return tokens, lemmas


def tf(word: str, doc: list[str]):
    # term frequency
    if len(doc) == 0:
        return 0
    return doc.count(word) / len(doc)


def idf(word: str, docs: list[list[str]]):
    # inverse document frequency
    docs_with_word = [word in doc for doc in docs].count(True)
    if len(docs) == 0:
        return 0
    if docs_with_word == 0:
        return 0
    return math.log2(len(docs) / docs_with_word)


def tf_idf(word: str, doc: list[str], docs: list[list[str]]):
    return tf(word, doc) * idf(word, docs)


def texts2docs(texts: list[str]):
    docs = []

    for text in texts:
        # Обработка текста, конвертация в список лемм
        _, doc = preprocess(text)
        docs += [doc]
    return docs


def get_features(doc: list[str], docs: list[list[str]]):
    # Document's words set
    uniques = set(doc)

    # Count TF-IDF for each word in document
    features = {}
    for unique in uniques:
        feature = tf_idf(unique, doc, docs)
        features[unique] = feature
    return features


def get_features_map(docs: list[list[str]]):
    return [get_features(doc, docs) for doc in docs]


def vectorize_doc(doc_features: dict, all_features: list):
    vector = []
    for feature in all_features:
        vector.append(doc_features.get(feature, 0.0))
    return np.array(vector)


def vectorize_docs(features_map: list[dict], all_features_names: list[str]):
    vectors = []
    for features in features_map:
        v = vectorize_doc(features, all_features_names)
        vectors.append(v)
    return np.vstack(vectors)


def vectorize_texts(train: list[str], test: list[str]):
    # Texts preprocessing
    train_size = len(train)
    test_size = len(test)

    texts = train[:] + test[:]
    docs = texts2docs(texts)

    # Create features set from all documents
    all_features_names_set = set()
    for doc in docs:
        all_features_names_set = all_features_names_set.union(set(doc))

    # Sort by name for document vectorization
    all_features_names = sorted(all_features_names_set)

    # Create features map - for each document: features and their TF-IDF
    features_map = get_features_map(docs)

    # Vectorize docs
    vectors = vectorize_docs(features_map, all_features_names)
    return vectors[:train_size], vectors[train_size:]


# Load datasets
texts_train, classes_train = load_dataset('train')
texts_test, classes_test = load_dataset('test')

# Convert classes to categorial (0 and 1)
classes_train = np.array([1 if x == 'A' else 0 for x in classes_train])
classes_test = np.array([1 if x == 'A' else 0 for x in classes_test])

docs_train, docs_test = vectorize_texts(texts_train, texts_test)

classifier = SVC()
classifier.fit(docs_train, classes_train)

predicted = classifier.predict(docs_test)

for i, filename in enumerate(os.listdir(os.path.join('texts', 'test'))):
    print(filename, predicted[i])

# Accuracy: (TP + TN) / (TP + FP + TN + FN)
print('Accuracy:', accuracy_score(classes_test, predicted))
# Precision: TP / (TP + FP)
print('Precision:', precision_score(classes_test, predicted))
# Recall: TP / (TP + FN)
print('Recall:', recall_score(classes_test, predicted))
# F1-score: 2 * precision * recall / (precision + recall)
print('F1-score:', f1_score(classes_test, predicted))
# Confusion matrix
positive, negative = confusion_matrix(classes_test, predicted)
TP, FP = positive
FN, TN = negative
print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')
