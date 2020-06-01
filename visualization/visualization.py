import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d

folder = 'data/test miou/'

x_list = []
losses = []
labels = []
for file in os.listdir(folder):
    df = pd.read_csv(folder+file)
    df.drop(['Wall time'], axis=1, inplace=True)
    x = df['Step'].to_list()
    loss = df['Value'].to_list()
    label = file.split('.')[0]
    x_list.append(x)
    losses.append(loss)
    labels.append(label)

for x, loss, label in zip(x_list, losses, labels):
    #from more_itertools import chunked
    #loss = [sum(loss) / len(loss) for loss in chunked(loss, 5)]
    #print(len(loss))
    x = x[::3]
    loss = loss[::3]
    #f1 = interp1d(x, loss, kind='cubic')
    #xnew = np.linspace(0, int(len(x)), num=100, endpoint=True)
    #xnew = xnew*5
    #plt.plot(xnew * 5, f1(xnew), label=label)
    plt.plot(x, loss, label=label)

plt.xlabel('Epoch')
plt.ylabel('mIoU')

#plt.title("Loss of Segmentation Model using Different Pretext Methods")
plt.legend(loc='upper left')
plt.show()
