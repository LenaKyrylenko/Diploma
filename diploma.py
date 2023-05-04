import numpy as np  #linear algebra
import pandas as pd # a data processing and CSV I/O library
# from pandas_profiling import ProfileReport
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv('./Covid Data.csv')
# df.head()
# df.info()

#1 -жінка 0 - чоловік, 1 - такб 0 - ні

# df['INTUBED'] = np.where(df['INTUBED'] == 97,2, df['INTUBED'])
# df['INTUBED'] = np.where(df['INTUBED'] == 99,1, df['INTUBED'])        
df['ICU'] = np.where(df['ICU'] == 97,2, df['ICU'])
df['ICU'] = np.where(df['ICU'] == 99,1, df['ICU'])   
def replace_values(val):
    if 1 <= val <= 3:
        return 1
    elif 4 <= val <= 7:
        return 0
    else:
        return val

df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].apply(replace_values)      
# df['PNEUMONIA'] = np.where(df['PNEUMONIA'] == 99,2,df['PNEUMONIA'])
# for i in('DIABETES','COPD', 'ASTHMA', 'INMSUPR',
#        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
#        'RENAL_CHRONIC', 'TOBACCO'):    
#     df =  df[(df[i] == '98')]     
    # df[i] =  np.where(df[i]== 98, 2, df[i])


# df['DEATH'] = [2 if row=='9999-99-99' else 1 for row in df['DATE_DIED']]

# список столбцов для замены
cols_to_replace = ['USMER', 'SEX','PATIENT_TYPE', 'DEATH', 'INTUBED',
                   'PNEUMONIA', 'PREGNANT', 
                   'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION',
                   'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY' , 'RENAL_CHRONIC',
                   
                   'TOBACCO', 'ICU'
                   ]

# заменить значения "2" на "0" в указанных столбцах
for col in cols_to_replace:
    df[col].replace(2, 0, inplace=True)



num_rows = len(df)
print('КОЛИЧЕСТВО ДО', num_rows)

# удаляем строки, где значение в столбцах "A" и "B" равны "удалить"
df = df[(df['DATE_DIED'] == '9999-99-99')]

# df = df[(df['PNEUMONIA'] != '99')]
num_rows = len(df)
print('КОЛИЧЕСТВО ПОСЛЕ', num_rows)



df = df.drop(columns=["DATE_DIED", "DEATH"])
def count_ones(column_name):
    return (df[column_name] == 1).sum()

pC1 = count_ones('CLASIFFICATION_FINAL')
pC2 = len(df) - count_ones('CLASIFFICATION_FINAL')
print("хворі = ",pC1)
print("здорові = ",pC2)
print(df['CLASIFFICATION_FINAL'][:100])


def check_age(age):
    if age >= 20 and age < 30:
        return "Возраст в пределах 20-30 лет"
    elif age >= 30 and age < 40:
        return "Возраст в пределах 30-40 лет"
    elif age >= 40 and age < 50:
        return "Возраст в пределах 40-50 лет"
    elif age >= 50 and age < 60:
        return "Возраст в пределах 50-60 лет"
    elif age >= 60 and age < 70:
        return "Возраст в пределах 60-70 лет"
    else:
        return "Возраст вне заданных пределов"
    
# def p_ones_age( p_covid):
#     count = 0 # переменная для хранения количества ячеек, содержащих значение 1
#     for i in range(1, len(df)): # цикл по всем строкам DataFrame
#         if df['AGE'][i] >= 20 and df['AGE'][i] <= 40 \
#         and df['CLASIFFICATION_FINAL'][i] == 1:
#             count += 1
#     print("count = ",count) # вывод результата
   
#     return (count)/p_covid    
# print("res ", p_ones_age(pC1))
def p_ones(column_name, p_covid):
    return ((df[column_name] == 1) & (df['CLASIFFICATION_FINAL'] == 1)).sum() / p_covid


print('ІМОВІРНІСТЬ ', p_ones('DIABETES', pC1))
# # разделение таблицы на независимые переменные (факторы) и зависимую переменную (целевую)
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # создание экземпляра PCA
# pca = PCA()

# # выполнение PCA на данных
# pca.fit(X)

# # получение важности признаков (столбцов)
# feature_importances = pca.explained_variance_ratio_

# # вывод важности признаков на экран
# for i, importance in enumerate(feature_importances):
#     print(f'Фактор {i+1}: {importance}')


# correlations = df.corr()

# for column in correlations:
#     # получаем 2 наибольших значения корреляции
#     nlargest = correlations[column].nlargest(2)
#     # проверяем, что количество элементов в серии больше 1
#     if len(nlargest) > 1:
#         second_largest = nlargest.iloc[-1]
#         print(f"biggest value in {column}:", second_largest)
#     else:
#         print(f"Not enough values for {column}")    

# # определяем столбец для корреляции
# corr_column = 'CLASIFFICATION_FINAL'

# # проходим по всем столбцам и вычисляем корреляцию с выбранным столбцом
# for column_name, column_data in df.iteritems():
#     if column_name != corr_column: 
#         corr = column_data.corr(df[corr_column])
#         print(f"Корреляция между столбцами '{column_name}' и '{corr_column}': {corr}")





# print(second_largest)
# print('correlations ', correlations)

# print("cor colomn ",  correlations.columns)
