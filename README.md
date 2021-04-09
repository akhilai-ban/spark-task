
# spark-task
grip


grip @ sparks foundation
data science and business analytics intern
task 1 predicting using supervised ml

# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline


# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

#displaying the data
data.head()

#checking the total data
data.shape

#checking information of given data
data.info()

#checking for duplicate
data.duplicated().sum()

#checking  null data persentage
data.isnull().sum()*100/len(data)

# Plotting the data to check relation bw them and distribution of data
data.plot(x='Hours', y='Scores', style='.',color='red')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X=data.drop(['Scores'],axis=1)
y=data['Scores']

#splitting the data into training and testing
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("model trained.")

model=LinearRegression()
model.fit(X_train,y_train)
line= model.coef_*X+model.intercept_

# plottting the trained data
plt.scatter(X,y,color= 'orange')
plt.plot(X,line,color='blue')
plt.xlabel('hours')
plt.ylabel('score')
plt.title('linear regression vs trained model')
plt.show()

print(X_test)
Y_pred=model.predict(X_test)

#comparing the  actual value with predicted value

df=pd.DataFrame({'Actual score' : y_test,'predicted  score ' : y_pred})

# predicting score for 9.25 hors of study

hours=9.25
pred=model.predict([[9.25]])
print("No of hours ={}".format(hours))
print("predicted score= {}".format(pred[0]))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#checking the efficiancy of model

mean_squ_error =mean_squared_error(y_test,y_pred[:5])
mean_abs_error = mean_absolute_error(y_test,y_pred[:5])

#printing the values

print("mean absolute error :",mean_abs_error)
print('mean squared error :' ,mean_squ_error)   #used for how close the data are to the fitted regression line
print('r2 Score : ',r2_score)
