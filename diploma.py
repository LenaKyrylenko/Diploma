import numpy as np  #linear algebra
import pandas as pd # a data processing and CSV I/O library
# from pandas_profiling import ProfileReport
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./Covid Data.csv')
# df.head()
# df.info()

#1 -жінка 0 - чоловік, 1 - такб 0 - ні

df['INTUBED'] = np.where(df['INTUBED'] == 97,2, df['INTUBED'])
df['INTUBED'] = np.where(df['INTUBED'] == 99,1, df['INTUBED'])        
df['ICU'] = np.where(df['ICU'] == 97,2, df['ICU'])
df['ICU'] = np.where(df['ICU'] == 99,1, df['ICU'])    
df['PREGNANT'] = np.where(df['PREGNANT'] == 97, 2, df['PREGNANT'])
df['PREGNANT'] = np.where(df['PREGNANT'] == 98, 1, df['PREGNANT'])
df['PNEUMONIA'] = np.where(df['PNEUMONIA'] == 99,2,df['PNEUMONIA'])
for i in('DIABETES','COPD', 'ASTHMA', 'INMSUPR',
       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
       'RENAL_CHRONIC', 'TOBACCO'):         
    df[i] =  np.where(df[i]== 98, 2, df[i])


df['DEATH'] = [2 if row=='9999-99-99' else 1 for row in df['DATE_DIED']]

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


# df.to_csv('./Covid Data.csv', index=False)

#виведення 
# for i in ('USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DEATH', 'INTUBED',
#        'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
#        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
#        'RENAL_CHRONIC', 'TOBACCO', 'ICU'):
    # print(df[i].value_counts())



# вычислить корреляцию между всеми столбцами таблицы
# correlation_matrix = df.corr()

# # создать тепловую карту
# print('correlation_matrix ', correlation_matrix) 
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')




correlations = df.corr()
print('correlations ', correlations)

print("cor colomn ",  correlations.columns)
# получение индексов строк, где коэффициент корреляции равен 1.0
# indices_to_drop = set()
# for column in correlations.columns:
#     indices = correlations.index[correlations[column] == 1.0].tolist()
#     print('indices', indices)
#     indices_to_drop.update(set(indices))

# print('indices_to_drop ',indices_to_drop)
# # удаление строк с найденными индексами
# df = df.drop(df.index[indices_to_drop])

# df.to_csv('./Covid Data.csv', index=False)


# # получение коэффициентов корреляции между всеми парами столбцов
# correlations = df.corr()

# # получение индексов строк, где коэффициент корреляции равен 1.0
# indices_to_drop = set()
# for column in correlations.columns:
#     indices = correlations.index[correlations[column] == 1.0].tolist()
#     print('indices', indices)
#     indices_to_drop.update(set(indices))

# print('indices_to_drop ',indices_to_drop)

# # удаление строк с найденными индексами
# for index in indices_to_drop:
#     if index in df.index:
#         df = df.drop(index)
#         print('index ', index)
# df.to_csv('./Covid Data.csv', index=False)
correlations = df.corr()

print(correlations)
for column in correlations:
    second_largest = correlations[column].nlargest(2).iloc[-1]
    print(f"biggest value in {column}:", second_largest)

print(second_largest)
# print('correlations.max', correlations.max())
# print(' correlations.min', correlations.min())
# correlation = df['PATIENT_TYPE'].corr(df['CARDIOVASCULAR'])

# print('PATIENT_TYPE и CARDIOVASCULAR', correlation)
# sns.countplot(x= y)