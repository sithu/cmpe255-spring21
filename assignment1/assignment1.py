import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.seterr(divide='ignore', invalid='ignore')

name= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset_file = "housing.csv"
df=pd.read_csv(dataset_file, delim_whitespace=True, names=name)
df.head()

row,col=df.shape
print("DF has ",row, "rows and ", col, " columns")
df.info()

df.corr()


df.describe()
x = df['LSTAT'].to_numpy().reshape(row, 1)
y = df['MEDV'].to_numpy().reshape(row, 1)

print(x.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(x)
#y=scaler.fit(y)
x=scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 22)
model = LinearRegression()
model.fit(x_train, y_train)

prediction=model.predict(x_test)
Rsquared = r2_score(y_test, prediction)
RMSE = np.sqrt(mean_squared_error(y_test, prediction))
print("RMSE:",RMSE)
print("R^2: ",Rsquared)
print(x_train.shape)
print(y_train.shape)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt.plot(x_train,model.predict(x_train))


polynomial = PolynomialFeatures(degree=2)
model_2 = make_pipeline(polynomial,model)
model_2.fit(x_train,y_train)
prediction_2=model_2.predict(x_test)
Rsquared = r2_score(y_test, prediction_2)
RMSER = np.sqrt(mean_squared_error(y_test, prediction_2))
print("RMSE:",RMSE)
print("R^2: ",Rsquared)
plt.figure()
plt.scatter(x_train,y_train,s=15)
plt.plot(x_train,model_2.predict(x_train),color="r")
plt.title("Polynomial regression with degree "+str(2))
plt.show()

polynomial = PolynomialFeatures(degree=20)
model_2 = make_pipeline(polynomial,model)
model_2.fit(x_train,y_train)
prediction_2=model_2.predict(x_test)
Rsquared = r2_score(y_test, prediction_2)
RMSE = np.sqrt(mean_squared_error(y_test, prediction_2))
print("RMSE:",RMSE)
print("R^2: ",Rsquared)
plt.figure()
plt.scatter(x_train,y_train,s=15)
plt.plot(x_train,model_2.predict(x_train),color="r")
plt.title("Polynomial regression with degree 20 "+str(2))
plt.show()

x2 = df[['LSTAT','RM',"PTRATIO"]].to_numpy()
y2 = df[['MEDV']].to_numpy()


scaler = StandardScaler()
scaler.fit(x2)
#y=scaler.fit(y)
x2=scaler.transform(x2)

x_train, x_test, y_train, y_test = train_test_split(x2,y,test_size=0.2,random_state = 22)
model = LinearRegression()
model.fit(x_train, y_train)

prediction=model.predict(x_test)
Rsquared  = r2_score(y_test, prediction)
RMSE = np.sqrt(mean_squared_error(y_test, prediction))
print("RMSE:",RMSE)
print("R^2: ",Rsquared)
print(x_train.shape)
print(y_train.shape)

