import pandas as pd
import numpy as np
import os

split = 'test'
data_dir = '/media/lmy'
csv_dir = '../' + 'video_{}_filelist.csv'.format(split)

df = pd.read_csv(csv_dir)
for index, row in df.iterrows():
    path = df.loc[index, 'path']
    df.loc[index, 'path'] = data_dir + path

df.to_csv("video_"+split+"_filelist.csv", sep=',', index=False)
