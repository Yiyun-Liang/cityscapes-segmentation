import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os

def smooth(csv_path,weight=0.8):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    return save


folder = 'data/train loss/'

x_list = []
losses = []
labels = []

for file in os.listdir(folder):
    #df = pd.read_csv(folder+file)
    df = smooth(folder+file)
    x = df['Step'].to_list()
    loss = df['Value'].to_list()
    label = file.split('.')[0]
    x_list.append(x)
    losses.append(loss)
    labels.append(label)

for x, loss, label in zip(x_list, losses, labels):
    plt.plot(x, loss, label=label)

plt.xlabel('Epoch')
plt.ylabel('Loss')

#plt.title("Loss of Segmentation Model using Different Pretext Methods")
plt.legend(loc='upper left')
plt.show()
