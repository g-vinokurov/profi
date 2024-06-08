# Profi

Программа классифицирует фрагменты текстов, содержащие профессиональный сленг.

Классы: A - сленг программистов, B - сленг финансистов, банкиров, экономистов

### Этапы работы:

#### 1. Предобработка текстов:
- приведение текста к одному регистру
- токенизация (разбиение на слова и знаки препинания, удаление пробелов)
- удаление знаков препинания
- удаление стоп-слов (слов, не несущих смысловой нагрузки)
- лемматизация (приведение слов к их нормальной форме для улучшения обучения)

#### 2. Отбор признаков для классификации - для каждого слова вычисляем его TF-IDF
  В реализации используются короткие фрагменты и метод опорных векторов,
  поэтому можно не уменьшать размерность пространства признаков

#### 3. Векторизация - для обучения с использованием метода опорных векторов,
  а также с помощью других возможных методов ML, требуется привести фрагменты
  к виду числовых векторов. Для этого существует несколько подходов:

  - Bag of Words - для каждого слова считаются его вхождения в документ.
    Компонента вектора соответствует слову из набора слов со всех документов.
    Общий плюс: быстрая векторизация документо относительно нейросетей.
    Некоторые из разновидностей:

    - простой подсчёт числа вхождений слова в документ
      компонента вектора - число вхождений
      - Плюсы: простота интерпертации
      - Минусы: не устойчивость для общеупотребительных слов - большие числа
      
    - "бинарый" мешок
      компонента вектора - 1 если слово есть в документе, 0 иначе
      - Плюсы: простота интерпретации, устойчивость к общеупотребительным словам
      - Минусы: теряется информация о количестве вхождений слова

    - использование TF-IDF метрики
      компонента вектора - значение метрики TF-IDF
      - Плюсы: устойчивость к общеупотребительным словам,
      сохранение информации о количестве вхождений
      
      TF (term frequency) - оценка важности слова в пределах одного документа

      TF = кол-во вхождений слова в документ / число слов в документе

      IDF (inverse document frequency) - инверсия частоты,
        с которой слово встречается во всех документах,
        уменьшает вес общеупотребительных слов,
        сохраняя внутри информацию о кол-ве вхождений слова
      
      IDF = log(кол-во документов / число вхождений слова в данный документ)
      
      TF-IDF = TF * IDF

  - Word2vec - специальный инструмент, основанный на искусственных нейросетях.
    На вход ему поступает текст, для каждого слова составляется вектор.
    При составлении векторов учитывается контекст употребления слова
    - Плюсы: учет контекста употребления слов
    - Минусы: относительно медленная скорость работы из-за нейросетей

#### 4. Построение и обучение классификатора
Существует несколько возможных решений для классификации:

- полуавтоматический подход: алгоритмически на основе созданных правил
  - Плюсы: интуитивная простота
  - Минусы: низкая точность

- автоматический подход: использование методов машинного обучения
  
  Здесь есть также несколько вариантов:

  - вероятностный метод (Метод Байеса)
    
    Класс документа подбирается так, чтобы условная вероятность, что документ
    принадлежит данному классу была максимальной.
    Для вычисления условной вероятности используется теорема Байеса.
    Из-за большого количества признаков делают предположение о статистической
    независимости каждой из координат вектора и применяют метод максимального
    правдоподобия.
    - Плюсы:
      - высокая скорость работы
      - не требуется вся выборка сразу (обучение на каждом образце)
      - простая программная реализаци
      - простота интерпретации результатов
    - Минусы:
      - низкая точность
      - невозможность учитывать зависимоть результата от сочетания признаков

  - Метод k ближайших соседей

    Для данного документа вычисляется расстояние до других документов.
    Расстояние - некоторая метрика.
    Выбирается k ближайших по расстоянию к данному документов.
    Классификация основана на том, что документ имеет класс, что и его соседи.

    - Плюсы:
      - можно не переобучать классификатор при обновлении выборки
      - алгоритм устойчив к аномальным выбросам данных
      - простая программная реализация
      - результаты легко поддаются интерпретации
      - хорошо работает на нелинейном разбиении объектов на классы
    - Минусы:
      - зависимость классификации от выбранной метрики
      - большая длительность работы из-за перебора всей выборки
      - не подходит для задач большой размерности

  - Деревья решений

    Чтобы классифицировать документ, мы идём от корня дерева к листам.
    В листах - наши категории.
    В других вершинах и корне - предикаты, по которым мы определяем ветвление.
    Существуют различные алгоритмы построения деревьев решений:
    - ID3
    - C4.5
    - CART
    
    В их основе лежит рекурсивное построение дерева и выбор предиката
    на основе некоторой метрики (н-р, прироста информации).
    
    - Плюсы:
      - простая программная реализация
      - простота интерпретации результатов
    - Минусы:
      - неустойчивость к большим выбросам исходных данных
      - лучше всего работает с дискретными признаками
      - требует больших объёмов данных для обучения

  - Метод опорных векторов

    Если выборка линейно разделяемая, то можно найти гиперплоскость
    (обычная плоскость в двумерном случае), которая разделяет лучше всего её
    Если выборка линейно неразделима, то можно скалярное произведение,
    которое используется в алгоритме, заменить функций-ядром
    Плоскость подбирается так, чтобы расстояние от неё до разделяемых множеств
    было максмальным.
    Под расстоянием до множеств имеется в виду расстояние
    до ближайших элементов этих множеств.
    - Плюсы:
      - высокая точность
      - достаточно небольшого набора для обучения
    - Минусы:
      - неустойчивость к выбросам
      - низкая скорость обучения

  - Искусственные нейронные сети

    Существуют различные архитектуры нейросетей.
    Для обработки текстов могут подойти RNN и CNN.
    RNN - рекуррентная нейросеть - хорошо подходит для обработки текстов,
    т.к. создана для обработки последовательностей и может учитывать контекст.
    CNN - сверточная нейросеть - в основном применяется для обработки картинок
    Но в нашей задаче нам не всегда нужен контекст всего фрагмента.
    Достаточно идти по тексту некоторым "окном" - свёрткой.
    - Плюсы:
      - высокая точность
      - может выявлять любые зависимости
      - может обучаться в реальном времени (с приходом новых образцов)
    - Минусы:
      - может не сойтись к решению и делать это долго
      - требует большого объема данных для обучения
      - низкая скорость обучения
      - сложная интерпретация и подбор параметров обучения

##### В результате было принято решение остановиться на методе опорных векторов, так как довольно трудно найти достаточно примеров, содержащих профессиональный сленг, а также из-за высокой точности

#### 5. Оценка эффективности классификации
Существуют разлиные метрики. Большинство основано на соотношениях между четырмя характеристиками:
##### TP - True Positive - число верно положительно классифицированных элементов
##### TN - True Negative - число верно отрицательно классифицированных элементов
##### FP - False Positive - число ложно-положительных результатов
##### FN - False Negative - число ложно-отрицательных результатов

##### Accuracy = (TP + TN) / (TP + FP + TN + FN)
Простая в плане интерпретации метрика, но плоха для несбалансированных классов. Чем ближе к 1, тем лучше

##### Precision = TP / (TP + FP)
Доля предсказанных положительных результатов, действительно положительных
Чем ближе к 1, тем лучше

##### Recall = TP / (TP + FN)
Доля реальных положительных результатов, предсказаных верно. Чем ближе к 1, тем лучше

##### F1-score = 2 * precision * recall / (precision + recall)
Специальная метрика, объединяющая precision и recall. Чем ближе к 1, тем лучше

##### Confusion matrix
Отражает FP, TP, FN, TN в удобной форме
