import numpy as np
import pandas as pd

path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/allData.csv"
data = pd.read_csv(path)

data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

data.head()

hair_growth_cols = [
    'Hair growth on Chin','Hair growth  on Cheeks','Hair growth Between breasts','Hair growth  on Upper lips', 
    'Hair growth in Arms', 'Hair growth on Inner thighs'
]

def label5(value):
    value = value.strip() if isinstance(value, str) else value
    if value == 'normal':
        return 0
    elif value == 'moderate':
        return 1
    else:
        return 2  
    
for col in hair_growth_cols:
    if col in data.columns:  
        data[col] = data[col].apply(label5)
    else:
        print(f"Warning: Column {col} not found in the dataset.")

data.head()

yes_no_cols = ['Overweight', 'loss weight gain / weight loss', 
    'irregular or missed periods', 'Difficulty in conceiving', 
    'Dark patches', 'always tired', 'more Mood Swings', 
    'canned food often', 'relocated city','Acne or skin tags','Hair thinning or hair loss']

def label16(value):
    if isinstance(value, str):  
        value = value.strip().lower()
        if value == 'yes':
            return 1
        else :
            return 0

for col in yes_no_cols:
    if col in data.columns: 
        data[col] = data[col].apply(label16)
    else:
        print(f"Warning: Column {col} not found in the dataset.")


data.head()

output_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data.csv"
data.to_csv(output_path, index=False)
print("data saved!")

data['Period Length'] = data['Period Length'].str.strip()
data['Cycle Length'] = data['Cycle Length'].str.strip()

def label17(row):
    if row['Period Length'] == '2-3 days':
        return 3
    elif row['Period Length'] == '4-5 days':
        return 5
    elif row['Period Length'] == '6-7 days':
        return 7
    else:
        return '9'

data['Period Length'] = data.apply(lambda row: label17(row), axis=1)
data.head()

def label18(row):
    if row['Cycle Length'] == '20-24 days':
        return 22
    elif row['Cycle Length'] == '20-28 days':
        return 25
    elif row['Cycle Length'] == '25-28':
        return 27
    elif row['Cycle Length'] == '29-35 days':
        return 32
    elif row['Cycle Length'] == '36+ days':
        return 37
    else:
        return 'NaN'

data['Cycle Length'] = data.apply(lambda row: label18(row), axis=1)
data.head()

del data['PCOS_from']

data

data.to_csv('C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_final.csv', index = False)
print("data_final saved!")

path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data.csv"
data1 = pd.read_csv(path)

data1['Period Length'] = data1['Period Length'].str.strip()
data1.head()

def label17(row):
    if row['Period Length'] == '2-3 days':
        return 3
    elif row['Period Length'] == '4-5 days':
        return 5
    elif row['Period Length'] == '6-7 days':
        return 7
    else:
        return 9

data1['Period Length'] = data1.apply(lambda row: label17(row), axis=1)

data1['Cycle Length'] = data1['Cycle Length'].str.strip()
data1.head()

def label18(row):
    
    if row['Cycle Length'] == '20-24 days':
        return 1
    elif row['Cycle Length'] == '20-28 days':
        return 2
    elif row['Cycle Length'] == '25-28 days':
        return 3
    elif row['Cycle Length'] == '29-35 days':
        return 4
    elif row['Cycle Length'] == '36+ days':
        return 5
    else:
        return 6

data1['Cycle Length'] = data1.apply(lambda row: label18(row), axis=1)


data1['Age'] = data1['Age'].str.strip()
data1.head()

def label19(row):
    if row['Age'] == 'Below 18':
        return 1
    elif row['Age'] == '18-25':
        return 2
    elif row['Age'] == '26-30':
        return 3
    elif row['Age'] == '31-35':
        return 4
    elif row['Age'] == '36-40':
        return 5
    elif row['Age'] == '41-45':
        return 6
    else:
        return 7

data1['Age'] = data1.apply(lambda row: label19(row), axis=1)

data1.head()
print(data1.head())
output_path = "C:/Users/hp/Desktop/Project/PCOS Prediction/Dataset/data_cleaned.csv"
data1.to_csv(output_path, index=False)
print("data_cleaned saved!")

