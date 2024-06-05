import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim

from pymorphy2 import MorphAnalyzer

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


def filter_features(doc: list[str], docs: list[list[str]], lim: int = -1):
    # Множество слов в документе
    features = set(doc)

    # Для каждого слова из множества считаем TF-IDF
    data = []
    for feature in features:
        weight = tf_idf(feature, doc, docs)
        data += [(feature, weight)]
    data.sort(key=lambda x: x[1], reverse=True)

    if lim >= 0:
        data = data[:min(lim, len(data))]

    features = [feature for feature, weight in data]
    return features


def create_features_map(docs: list[list[str]], features_lim: int = -1):
    # "Карта" отфильтрованных признаков
    features_map = []
    for doc in docs:
        features = filter_features(doc, docs, features_lim)
        features_map += [features]
    return features_map


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


docs = texts2docs([TEXT1, TEXT2])
features_map = create_features_map(docs, 50)
