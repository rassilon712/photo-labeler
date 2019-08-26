import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("./batch_data.csv")
x = data[data.columns[0]]
y = data['time']
plt.title("time per batch")
plt.legend(loc='upper left', frameon=True)
plt.ylabel('time')
plt.xlabel('batch')
plt.plot(x,y, label='user5')
