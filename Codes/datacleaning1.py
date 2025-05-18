import pandas as pd
import numpy as np

copied_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/results.csv"
data = pd.read_csv(copied_path)
# print(data.columns)
data.columns = data.columns.str.strip()

data.head()

data.drop('Timestamp', inplace=True, axis=1)
data.drop('PCOS tested', inplace=True, axis=1)
data.drop('When do you experience mood swings?', inplace=True, axis=1)

data.head()

data["City"] = data["City"].str.lower()  # lower all city names

data.head()

data = data.rename(columns={'PCOS from age of': 'PCOS_from'})

data['PCOS_from'] = data.PCOS_from.str.extract(r'(\d+)')

data.head()

data.to_csv('C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/allData.csv', index=False)

PCOS_True = data[data['PCOS_from'].notnull()]


PCOS_True = PCOS_True.dropna(subset=["PCOS_from"])


PCOS_True.head()

PCOS_True.to_csv('C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/onlyPCOS.csv', index=False)

PCOS_False = data[data['PCOS_from'].isnull()]
PCOS_False.to_csv('C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/NoPCOS.csv', index=False)

print("allData, onlyPCOS and NoPCOS saved to dataset folder")