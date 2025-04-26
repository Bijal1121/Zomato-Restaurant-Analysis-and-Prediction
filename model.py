import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomato_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
x=df.drop('Rating',axis=1)
y=df['Rating']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


#Preparing random forest regressor
RF = RandomForestRegressor()
RF.fit(x_train,y_train)


y_predict=RF.predict(x_test)


import pickle
# # Saving model to disk
pickle.dump(RF, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)