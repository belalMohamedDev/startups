import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
np.random.seed(42)

#read data
ReadData=pd.read_csv(r"E:\بايثون\P14-Part2-Regression\Section 7 - Multiple Linear Regression\Python\50_Startups.csv")

#show sample from data
print(ReadData.head())
print("--------------------------------")
print (ReadData.tail())
print("-------------------------------")

#show missing data
print(ReadData.info())

# get all the unique values in the 'state' column
state = ReadData['State'].unique()
state.sort()
print(state)

#convert data in coluom state and remove empty space
ReadData["State"]=ReadData["State"].str.lower()
ReadData["State"]=ReadData["State"].str.strip()

# get all the unique values in the 'state' column
state = ReadData['State'].unique()
state.sort()
print(state)

#map data 
x=ReadData.drop("Profit",axis=1)
y=ReadData["Profit"]

#encoder data 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(x))


#split data
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#choose model 
model=LinearRegression()

#train model 
model.fit(x_train,y_train)

#print best train and test in model

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

#save model 
pickle.dump(model,open(r"statrtups_model.pkl","wb"))