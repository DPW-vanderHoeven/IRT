#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read (import) the dataset
df=pd.read_csv(r'C:\Users\Daan\Documents\Individual Research Training - ANN\Data\IRT_Daan.csv')

#visualizing the data using heatmap
plt.figure()
sns.heatmap(df.corr(),cmap='coolwarm')
plt.show()

#selecting the parameters
price = df["lnsalesprice"]
rooms = df["rooms"]
land = df["land"]
pool = df["dpool"]
year_2019 = df["Year_2019"]
year_2018 = df["Year_2018"]
year_2017 = df["Year_2017"]
year_2016 = df["Year_2016"]