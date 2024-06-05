import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from pymorphy2 import MorphAnalyzer

import math

STOP_WORDS = set(stopwords.words('russian'))
PUNCTUATIONS = ',.;:\'\"[]()?!'


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
    print(tokens)

    # Удаление знаков препинания
    tokens = [x for x in tokens if x not in PUNCTUATIONS]
    print(tokens)

    # Лемматизация - приведение слов в их нормальную форму
    morph = MorphAnalyzer()
    lemmas = [(x, morph.normal_forms(x)) for x in tokens]
    lemmas = [normals[0] if normals else x for x, normals in lemmas]
    print(lemmas)

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
    if len(docs_with_word) == 0:
        return 0
    return math.log2(len(docs) / docs_with_word)


def tf_idf(word: str, doc: list[str], docs: list[list[str]]):
    return tf(word, doc) * idf(word, docs)


TEXT = '''Идейные соображения высшего порядка, а также дальнейшее развитие
 различных форм деятельности представляет собой интересный эксперимент проверки 
 модели развития. Идейные соображения высшего порядка, 
 а также укрепление и развитие структуры позволяет оценить значение дальнейших 
 направлений развития. Таким образом сложившаяся структура организации играет 
 важную роль в формировании существенных финансовых и административных условий.
'''

tokens, lemmas = preprocess(TEXT)

# Формирование биграмм
bigrams = list(ngrams(lemmas, 2))
print(bigrams)
