# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Start the program

Step 2:Import the required library and read the dataframe.

Step 3:Write a function computeCost to generate the cost function.

Step 4:Perform iterations og gradient steps with learning rate.

Step 5:Plot the Cost function using Gradient Descent and generate the required graph.

Step 6:End the program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ADITAAYAN M
RegisterNumber: 212223040006
*/
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta= np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predicitions= (X).dot(theta).reshape(-1,1)
        errors=(predicitions-y).reshape(-1,1)
        theta-= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data = pd.read_csv("C:/Users/admin/OneDrive/Desktop/50_Startups.csv")
print (data.head())
X=(data.iloc[1:,:-2].values)
X1= X.astype(float)
s= StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled= s.fit_transform(X1)
y1_scaled= s.fit_transform(y)
print(X)
print(X1_scaled)
print(y1_scaled)
theta=linear_regression(X1_scaled,y1_scaled)
new_Data= np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled= s.fit_transform(new_Data)
predicition= np.dot(np.append(1,new_scaled),theta)
predicition= predicition.reshape(-1,1)
pre= s.inverse_transform(predicition)
print(predicition)
print(f"Predicited value: {pre}")
```

## Output:
![363573299-4da86d90-beef-48c1-af1c-732d1d44e53a](https://github.com/user-attachments/assets/6705f8e2-aef2-4fdb-a7f1-dc7c7ceed7dd)

![363573319-13552e56-0a84-46df-b24d-a7c6a968e36e](https://github.com/user-attachments/assets/9c3ded16-08e0-4e34-a681-18385ad3b057)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
