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
    # Если нормальной формы не нашлось, оставляем как есть
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
    if docs_with_word == 0:
        return 0
    return math.log2(len(docs) / docs_with_word)


def tf_idf(word: str, doc: list[str], docs: list[list[str]]):
    return tf(word, doc) * idf(word, docs)


def analyze_texts(texts: list[str]):
    docs = []
    for text in texts:
        # Обработка текста, конвертация в список лемм
        _, doc = preprocess(text)
        docs += [doc]

    for doc in docs:
        # Множество слов в документе
        uniques = set(doc)

        # Для каждого слова из множества считаем TF-IDF
        weigths = {}

        for unique in uniques:
            weigths[unique] = tf_idf(unique, doc, docs)
        print(weigths)


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


analyze_texts([TEXT1, TEXT2])
