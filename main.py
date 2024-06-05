import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

LANGUAGE = 'russian'

STOP_WORDS = set(stopwords.words(LANGUAGE))


def preprocess(text: str):
    # Приводим к одному регистру
    text = text.lower()

    # Разбиваем текст на токены (знаки пунктуации, слова)
    tokens = word_tokenize(text, language=LANGUAGE)

    # Убираем стоп-слова (слова без смысловой нагрузки)
    tokens = [x for x in tokens if x not in STOP_WORDS]

    # Не работает нормально на русском
    # from nltk.stem.snowball import SnowballStemmer
    # stemmer = SnowballStemmer(language=LANGUAGE)
    # stems = [stemmer.stem(x) for x in tokens]
    print(tokens)


TEXT = '''Идейные соображения высшего порядка, а также дальнейшее развитие
 различных форм деятельности представляет собой интересный эксперимент проверки 
 модели развития. Идейные соображения высшего порядка, 
 а также укрепление и развитие структуры позволяет оценить значение дальнейших 
 направлений развития. Таким образом сложившаяся структура организации играет 
 важную роль в формировании существенных финансовых и административных условий.
'''


preprocess(TEXT)

