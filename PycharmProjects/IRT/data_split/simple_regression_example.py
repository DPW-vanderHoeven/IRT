#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read (import) the dataset
df=pd.read_csv(r'C:\Users\Daan\Documents\Individual Research Training - ANN\Data\IRT_Daan.csv')

#visualizing the data using heatmap
plt.figure()
sns.heatmap(df.corr(),cmap='coolwarm')
plt.show()

#selecting the parameters
price = df["lnsalesprice"]
rooms = df["rooms"]

x = np.array(rooms).reshape(-1, 1)
y = np.array(price)

#split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#fit the model over the training dataset
model = LinearRegression()
model.fit(X_train, y_train)

#calculate intercept and coefficient
print(model.intercept_)
print(model.coef_)

pred = model.predict(X_test)
predictions = pred.reshape(-1, 1)

#calculate RMSE to evaluate model performance
print('MSE: ', mean_squared_error(y_test,predictions))
print('RMSE: ', np.sqrt(mean_squared_error(y_test,predictions)))
