# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
data = pd.read_csv('/content/russia_losses_equipment.csv')
data['date'] = pd.to_datetime(data['date'])
X = np.array(data.index).reshape(-1, 1)
y = data['tank']
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)
plt.figure(figsize=(35, 5))
plt.subplot(1,3,1)
plt.plot(data['date'], data['tank'], label='Tank Losses')
plt.xlabel('Date')
plt.ylabel('Tank Losses')
plt.title('Year-wise Tank Losses Over Time')
plt.grid(True)

plt.figure(figsize=(35, 5))
plt.subplot(1,3,2)
plt.plot(data['date'], y, label='Actual Tank Losses')
plt.plot(data['date'], y_pred_linear, color='red',linestyle='--', label='Linear Trend')
plt.xlabel('Date')
plt.ylabel('Tank Losses')
plt.title('Linear Trend Estimation for Tank Losses')
plt.legend()
plt.grid(True)


plt.figure(figsize=(35, 5))
plt.subplot(1,3,3)
plt.plot(data['date'], y, label='Actual Tank Losses')
plt.plot(data['date'], y_pred_poly, color='green',linestyle='--', label='Polynomial Trend (Degree 2)')
plt.xlabel('Date')
plt.ylabel('Tank Losses')
plt.title('Polynomial Trend Estimation for Tank Losses')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
<img width="1042" height="556" alt="image" src="https://github.com/user-attachments/assets/df47383b-180c-46f3-b91a-a80fac89ba14" />

A - LINEAR TREND ESTIMATION
<img width="1042" height="558" alt="image" src="https://github.com/user-attachments/assets/940ecc0d-73f4-4edc-ab10-213ef7e2c751" />



B- POLYNOMIAL TREND ESTIMATION
<img width="1044" height="555" alt="image" src="https://github.com/user-attachments/assets/cb127a37-baf9-4769-9d54-cea2b4588575" />



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
