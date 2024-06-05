import nltk

nltk.download('punkt')
nltk.download('stopwords')

import gensim

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pymorphy2 import MorphAnalyzer

import dataclasses
import math

STOP_WORDS = set(stopwords.words('russian'))
PUNCTUATIONS = ',.;:\'\"[]()?!-'


def preprocess(text: str):
    # Приводим к одному регистру
    text = text.lower()

    # Разбиваем текст на токены (знаки пунктуации, слова)
    tokens = word_tokenize(text, language='russian')

    # Убираем стоп-слова (слова без смысловой нагрузки)
    tokens = [x for x in tokens if x not in STOP_WORDS]

    # Не работает нормально на русском
    # from nltk.stem.snowball import SnowballStemmer
    # stemmer = SnowballStemmer(language=LANGUAGE)
    # stems = [stemmer.stem(x) for x in tokens]

    # Удаление знаков препинания
    tokens = [x for x in tokens if x not in PUNCTUATIONS]

    # Лемматизация - приведение слов в их нормальную форму
    morph = MorphAnalyzer()
    lemmas = [(x, morph.normal_forms(x)) for x in tokens]
    # Если нормальной формы не нашлось, оставляем как есть
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
    # Множество слов в документе
    uniques = set(doc)

    # Для каждого слова из множества считаем TF-IDF
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
    return tuple(vector)


def vectorize_docs(features_map: list[dict], all_features_names: list[str]):
    vectors = []
    for features in features_map:
        v = vectorize_doc(features, all_features_names)
        vectors.append(v)
    return vectors


TEXT1 = '''Идейные соображения высшего порядка, а также дальнейшее развитие
 различных форм деятельности представляет собой интересный эксперимент проверки 
 модели развития. Идейные соображения высшего порядка, 
 а также укрепление и развитие структуры позволяет оценить значение дальнейших 
 направлений развития. Таким образом сложившаяся структура организации играет 
 важную роль в формировании существенных финансовых и административных условий.
'''

TEXT2 = '''Не следует, однако забывать, что консультация с широким активом в 
значительной степени обуславливает создание позиций, занимаемых участниками в 
отношении поставленных задач. Товарищи! постоянное 
информационно-пропагандистское обеспечение нашей деятельности обеспечивает 
широкому кругу (специалистов) участие в формировании соответствующий условий 
активизации. Идейные соображения высшего порядка, а также рамки и место 
обучения кадров позволяет оценить значение соответствующий условий активизации. 
Значимость этих проблем настолько очевидна, что консультация с широким активом 
в значительной степени обуславливает создание новых предложений.
'''


# Предобработка текстов
docs = texts2docs([TEXT1, TEXT2])

# Формируем множество признаков из всех документов
all_features_names_set = set()
for doc in docs:
    all_features_names_set = all_features_names_set.union(set(doc))

# Упорядочиваем для последующей векторизации документов
all_features_names = sorted(all_features_names_set)

# Формируем карту признаков - для каждого документа его признаки с TF-IDF
features_map = get_features_map(docs)

# Векторизуем документы
vectors = vectorize_docs(features_map, all_features_names)
for vector in vectors:
    print(vector)

