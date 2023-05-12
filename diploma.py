import numpy as np  #linear algebra
import pandas as pd # a data processing and CSV I/O library
# from pandas_profiling import ProfileReport
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,r2_score,f1_score
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from functools import reduce
# df.head()
# df.info()

#1 -жінка 0 - чоловік, 1 - такб 0 - ні

# df['INTUBED'] = np.where(df['INTUBED'] == 97,2, df['INTUBED'])
# df['INTUBED'] = np.where(df['INTUBED'] == 99,1, df['INTUBED'])        
# df['ICU'] = np.where(df['ICU'] == 97,2, df['ICU'])
# df['ICU'] = np.where(df['ICU'] == 99,1, df['ICU'])   


def replace_values(val):
    if val >= 1 and val <= 3:
        return 1
    elif  val >=4 and val <= 7:
        return 0
    else:
        return val

def clear_data(df):
    #заменяем значение в столбце классификация 
    df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].apply(replace_values)      
    # список столбцов для замены
    cols_to_replace = ['USMER', 'SEX','PATIENT_TYPE',
                       'PNEUMONIA', 
                       'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION',
                       'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY' , 'RENAL_CHRONIC',
                       'TOBACCO',
                       ]
     # заменить значения "2" на "0" в указанных столбцах
    for col in cols_to_replace:
        df[col].replace(2, 0, inplace=True)
    num_rows = len(df)
    #удаление рядков с 
    
    df = df.loc[df['DATE_DIED'] == '9999-99-99']
    # num_rows = len(df)

    #удаление столбцов о дате сметри и смерти
    df = df.drop(columns=["DATE_DIED", "PREGNANT", "INTUBED", "ICU"])
    for col in df.columns:
        condition = df[(df[col] == 97) | (df[col] == 98) | (df[col] == 99)]
        df.drop(condition.index,inplace=True)
    df.to_csv('updated_table.csv', index=False)

# def shannon_entropy(labels):
#     """Вычисляет информационную энтропию для массива меток."""
#     n_labels = len(labels)
#     print("n label ", n_labels)
#     if n_labels <= 1:
#         return 0
#     counts = np.bincount(labels)
#     print("count ", counts)
#     probs = counts / n_labels
#     print("probs ", probs)
#     n_classes = np.count_nonzero(probs)
#     if n_classes <= 1:
#         return 0
#     ent = 0.
#     # Вычисляем энтропию по формуле Шеннона
#     for i in probs:
#         ent -= i * np.log2(i)
#     return ent

def information_gain(feature, labels):
    """Вычисляет информационный выигрыш для данного признака."""
    # Разбиваем на две группы
    classes = np.unique(labels)
    n_instances = len(labels)
    ent = shannon_entropy(labels)
    new_ent = 0.
    for cl in classes:
        mask = feature == cl
        sub_labels = labels[mask]
        sub_ent = shannon_entropy(sub_labels)
        new_ent += len(sub_labels) / n_instances * sub_ent
    # Вычисляем информационный выигрыш
    info_gain = ent - new_ent
    return info_gain

# Загрузим таблицу
#МЕТОД ШЕННОНА

def method_shennon():
# Выделим признаки и метки
    features = df.drop('CLASIFFICATION_FINAL', axis=1)
    labels = df['CLASIFFICATION_FINAL']
    
    # Вычислим информационный выигрыш для каждого признака
    info_gains = []
    for col in features.columns:
        feature = features[col]
        info_gain = information_gain(feature, labels)
        
        info_gains.append(info_gain)
    # Отсортируем признаки по убыванию информационного выигрыша
    sorted_idx = np.argsort(info_gains)[::-1]

    sorted_features = features.columns[sorted_idx]

    # Выведем признаки и их информационный выигрыш
    for feature in sorted_features:
        print("признак: {:<15} \t информативность: {}".format(feature,information_gain(features[feature], labels)))

def shannon_entropy(data, feature):
    # вычисление общего количества объектов
    total_count = data.shape[0]
    # вычисление количества уникальных значений признака
    unique_vals = data[feature].unique()
    # вычисление количества объектов для каждого уникального значения признака
    val_counts = data[feature].value_counts()
    # вычисление вероятности каждого уникального значения признака
    val_probs = val_counts / total_count
    # вычисление меры Шеннона
    shannon = -np.sum(val_probs * np.log2(val_probs))
    return shannon

def method_shennon2(df):
# Выделим признаки и метки
    target = df['CLASIFFICATION_FINAL']
    features = df.drop('CLASIFFICATION_FINAL', axis=1)
    
    for col in features.columns:
        feature = features[col]
       
    # features = ['Age', 'Ширина почки', 'Толщина', 'Поренхима','Ускорение']
        age_entropy = shannon_entropy(df, col)
        
        print("Entropy of ", col, " : ", age_entropy)
    # features = df.drop('CLASIFFICATION_FINAL', axis=1)
    # labels = df['CLASIFFICATION_FINAL']
    
    # Вычислим информационный выигрыш для каждого признака
    
def count_ones(column_name):
    return (df[column_name] == 1).sum()

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

def feature_information_gain(X, y, feature):
    pivot_table = pd.pivot_table(
        pd.DataFrame({'X': X, 'y': y}), 
        values='X', 
        index='y', 
        aggfunc=len
    ).fillna(0)
    
    values = pivot_table.index.tolist()
    counts = pivot_table['X'].tolist()
    total_count = sum(counts)
    
    feature_entropy = sum(
        [(counts[i] / total_count) * entropy(X[y == values[i]]) 
         for i in range(len(values))]
    )
    
    info_gain = entropy(X) - feature_entropy
    return info_gain

def check_shennon():    
        # список всех признаков, за исключением целевой переменной
    features = list(df.columns[:-1])
    
    # создаем словарь, чтобы хранить информативность каждого признака
    info_gain_dict = {}
    
    # расчет информативности для каждого признака
    for feature in features:
        ig = information_gain(df, feature, 'CLASIFFICATION_FINAL')
        info_gain_dict[feature] = ig
    
    # выводим значения информативности каждого признака
    for feature, ig in info_gain_dict.items():
        print(f'Information Gain for {feature}: {ig}')
    age_entropy = shannon_entropy(df, 'Age')
    print("Entropy of age: ", age_entropy)
    age_entropy = shannon_entropy(df, 'Длина почки')
    print("Entropy of Длина почки: ", age_entropy)
    age_entropy = shannon_entropy(df, 'Ширина почки')
    print("Entropy of Ширина почки: ", age_entropy)
    age_entropy = shannon_entropy(df, 'Толщина')
    print("Entropy of Толщина: ", age_entropy)
    age_entropy = shannon_entropy(df, 'Поренхима')
    print("Entropy of Поренхима: ", age_entropy)
    age_entropy = shannon_entropy(df, 'Ускорение')
    print("Entropy of Ускорение: ", age_entropy)
    
# target = 'Длина почки'
# features = ['Age', 'Ширина почки', 'Толщина', 'Поренхима','Ускорение']
# функция для расчета энтропии
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# функция для расчета информативности признака
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[feature]==values[i]).dropna()[target]) for i in range(len(values))])
    info_gain = total_entropy - weighted_entropy
    return info_gain

pd.options.mode.chained_assignment = None  # default='warn'
# df = pd.read_excel('data1.xlsx')
df = pd.read_csv('updated_table.csv')

# clear_data(df)
for col in df.columns:
        condition = df[(df[col] == 97) | (df[col] == 98) | (df[col] == 99)]
        df.drop(condition.index,inplace=True)
        
for i in df.columns:
    print(df[i].value_counts())
    print("\n")
scaler = StandardScaler()
    
    # нормализуем данные
normalized_data = scaler.fit_transform(df)
column_names = df.columns.tolist()
    # создаем DataFrame из нормализованных данных
normalized_df = pd.DataFrame(normalized_data, columns=column_names)

# clear_data(df)
# for feature in features:
#     info_gain = feature_information_gain(df[feature], df[target], feature)
#     print(f'{feature}: {info_gain}')

# clear_data(df)
# method_shennon()
# method_shennon2(normalized_df)
print()
num_rows = len(df)
# print('КОЛИЧЕСТВО ', num_rows)
print("num_rows", num_rows)
count_pc1 = count_ones('CLASIFFICATION_FINAL')
count_pc2  = (len(df) - count_ones('CLASIFFICATION_FINAL'))
pC1 =count_ones('CLASIFFICATION_FINAL') / num_rows
pC2 = (len(df) - count_ones('CLASIFFICATION_FINAL'))/num_rows

print("хворі = ",pC1)
print("здорові = ",pC2)
# df = df.drop(columns=["USMER"])


def check_age(age,p_covid, pC1):
    if age >= 20 and age < 30:
        return  p_ones_age(20,30,p_covid,pC1)
    elif age >= 30 and age < 40:
        return  p_ones_age(30,40,p_covid,pC1)
    elif age >= 40 and age < 50:
        return  p_ones_age(40,50,p_covid,pC1)
    elif age >= 50 and age < 60:
        return  p_ones_age(50,60,p_covid,pC1)
    elif age >= 60 and age < 70:
        return  p_ones_age(60,70,p_covid,pC1)
    else:
        return "Возраст вне заданных пределов"
    
def p_ones_age(age_from, age_up_to,  p_covid,pc1):
    count = 0
    return ((df['AGE'] >= age_from) &(df['AGE'] <= age_up_to) & (df['CLASIFFICATION_FINAL'] == pc1))
       
def p_ones(column_name, p_covid):
    return ((df[column_name] == 1) & (df['CLASIFFICATION_FINAL'] == 1)).sum() / p_covid

def normalize_resultPC1(resC1, resC2 ):
    return resC1/(resC1+resC2)

def normalize_resultPC2(resC1, resC2 ):
    return resC2/(resC1+resC2)


def calc(pC1 = pC1, usmer = 1, medical_unit = 2, sex = 1, patient_type = 1, medical = 3, other = 0,  obesity=1,
          intubed = 1, pneumonia = 1, age = 50, diabetes = 1, copd = 1, astma = 1, tobacco = 0,pc1 = 1,  p_covid=count_pc1):
    res_usmer =  ((df['USMER'] == usmer) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_patient_type =  ((df['PATIENT_TYPE'] == patient_type) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_medical = ((df['MEDICAL_UNIT'] == medical) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_sex =  (((df['SEX'] == sex) & (df['CLASIFFICATION_FINAL'] == pc1)).sum()) / p_covid
    res_pneumonia = ((df['PNEUMONIA'] == pneumonia) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_age = check_age(age,p_covid, pc1)
    res_diabetes = ((df['DIABETES'] == diabetes) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_copd = ((df['COPD'] == copd) & (df['CLASIFFICATION_FINAL'] ==pc1)).sum() / p_covid
    res_astma = ((df['ASTHMA'] == astma) & (df['CLASIFFICATION_FINAL'] == pc1)).sum() / p_covid
    res_tobacco = ((df['TOBACCO'] == tobacco) & (df['CLASIFFICATION_FINAL'] ==pc1)).sum() / p_covid
    res_obesity = ((df['OBESITY'] == obesity) & (df['CLASIFFICATION_FINAL'] ==pc1)).sum() / p_covid
    # res = res_sex * res_medical * pC1
    
    resultPC1 =res_sex  * res_obesity * res_medical * res_pneumonia * res_age * res_diabetes * res_copd * res_astma * res_tobacco* pC1
    
    print("res_medical = ", res_medical)
    print("res_sex = ", res_sex)
    # print("res = ", res)
    # print("res_pneumonia = ", res_pneumonia)
    # print("res_age = ", res_age)
    # print("res_diabetes = ", res_diabetes)
    # print("res_copd = ", res_copd)
    # print("res_astma = ", res_copd)
    # print("res_tobacco = ", res_copd)
    
    return  resultPC1

print("Вхідні дані: \nsex = 1,intubed = 1, pneumonia = 1, age = 30, diabetes = 0, copd = 0, astma = 0, tobacco = 0")
# resC1 = calc(pc1 = 1)
# resC2 = calc(pC1= pC2, p_covid=count_pc2, pc1 = 0)

# print("імовірність захворіти = ", normalize_resultPC1(resC1, resC2))
# print("імовірність здоровий = ",normalize_resultPC2(resC1, resC2))

print("medical unit від 1 до 13")
print("usmer - чи лікувався в медичних закладах")
print("patient_type - 1 - лікується вдома, 0 госпіталізован")
print("INMSUPR - ослаблення імунітету, 1 - так, 0 - ні")
print("HIPERTENSION - підвищенний кров'яний тиск 1 - так, 0 - ні")
print("RENAL_CHRONIC - хронічна хвороба нирок")

user_input = pd.read_csv('user1.csv')
columns_to_check = ['USMER',  'MEDICAL_UNIT', 'SEX','PATIENT_TYPE', 'PNEUMONIA','AGE', 'DIABETES', 
                    'COPD', 'ASTHMA','INMSUPR', 'HIPERTENSION','OTHER_DISEASE',
                    'CARDIOVASCULAR', 'OBESITY','RENAL_CHRONIC',
                    'TOBACCO']
# for i in user_input.columns:
#     print(user_input[i].value_counts())
#     print("\n")
    
# print(df.columns)
# print(user_input.columns)

def probility_sick(pC1= pC1, user_input = user_input,pc1 = 1, p_covid = count_pc1):
    result_dict = {}
    for column, value in user_input.items():
        # print("column ", column)
        if column == 'AGE':
            age = int(user_input[column])
            relevant_data = check_age(age,p_covid, pc1)
        else:
            relevant_data = (df[column] == int(value)) & (df['CLASIFFICATION_FINAL'] == pc1)
        likelihood = relevant_data.sum() / p_covid
        result_dict[column] = likelihood
    result_dict.popitem() # видаляємо останній запис
    resultPC1 = pC1 * reduce(lambda x, y: x * y, result_dict.values())
    print()
    print(" результат до нормалізації =  ", resultPC1)
    print()
    print("result dict ", result_dict)
    return resultPC1
resC1 = probility_sick(pc1 = 1)
resC2 = probility_sick(pC1= pC2, p_covid=count_pc2, pc1 = 0)
print()
print("імовірність захворіти = ", normalize_resultPC1(resC1, resC2))
print("імовірність здоровий = ",normalize_resultPC2(resC1, resC2))

    # features = df.drop('CLASIFFICATION_FINAL', axis=1)
    # labels = df['CLASIFFICATION_FINAL']
    
    # # Вычислим информационный выигрыш для каждого признака
    # info_gains = []
    # for col in features.columns:
    #     feature = features[col]
    #     info_gain = information_gain(feature, labels)
    #     info_gains.append(info_gain)
    
    # # Отсортируем признаки по убыванию информационного выигрыша
    # sorted_idx = np.argsort(info_gains)[::-1]
    # sorted_features = features.columns[sorted_idx]
    
    # # Выведем признаки и их информационный выигрыш
    # for feature in sorted_features:
    #     print("признак: {:<15} \t информативность: {:.5f}".format(feature,information_gain(features[feature], labels)))


# method_kylbaka(df)

# print(correlation(df))
def correlation2(df):
    # создаем объект StandardScaler
    scaler = StandardScaler()
    
    # нормализуем данные
    normalized_data = scaler.fit_transform(df)
    column_names = df.columns.tolist()
    # создаем DataFrame из нормализованных данных
    normalized_df = pd.DataFrame(normalized_data, columns=column_names)
    
    # находим корреляцию
    # corr_matrix = normalized_df.corr()
    correlations = normalized_df.corr()
    
    for column in correlations:
        # получаем 2 наибольших значения корреляции
        nlargest = correlations[column].nlargest(2)
        nsmallest = correlations[column].nsmallest(1)
        # print("nsmallest ", nsmallest)
        # проверяем, что количество элементов в серии больше 1
        if len(nlargest) > 1:
            second_largest = nlargest.iloc[-1]
            
            print("{:<15} {:.5} {:.5}".format(column, second_largest,nsmallest.item()))
        else:
            print(f"Not enough values for {column}")    
    
    # определяем столбец для корреляции
    corr_column = 'CLASIFFICATION_FINAL'
    
    # проходим по всем столбцам и вычисляем корреляцию с выбранным столбцом
    for column_name, column_data in normalized_df.iteritems():
        if column_name != corr_column: 
            corr = column_data.corr(df[corr_column])
            print(f"Корреляция между столбцами '{column_name}' и '{corr_column}': {corr}")
    
# correlation2(df)
# print("Кореляція")
# correlation2()


# print(second_largest)
# print('correlations ', correlations)

# print("cor colomn ",  correlations.columns)
