#Data collection
import pandas as pd
data=pd.read_csv('USA_Housing.csv')

#Data Cleaning
data=data.drop("Address",axis=1)

#Information about Target variable
data.Price.describe()

#Visualization
import seaborn
seaborn.pairplot(data)
seaborn.displot(data)


#Creating Arrays {feature & target}
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#Splitting the data set
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.1,random_state=10)


#Algorithm selection and Training  
'''
Sklearn
linear model
Algorithm-Linear Regression
'''
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(Xtrain, Ytrain)
Prediction=lin.predict(Xtest)
Accuracy1=lin.score(Xtest, Ytest)


'''
Sklearn
linear model
Algorithm-BayesianRidge
'''
from sklearn.linear_model import BayesianRidge 
BR=BayesianRidge()
BR.fit(Xtrain,Ytrain)
Accuracy2=BR.score(Xtest, Ytest)


'''
Sklearn
svm
Algorithm-SVR
'''
from sklearn.svm import SVR
s=SVR()
s.fit(Xtrain,Ytrain)
Accuracy3=s.score(Xtest, Ytest)


