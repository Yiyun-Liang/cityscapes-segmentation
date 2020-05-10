import pandas as pd
import numpy as np
import os


data_dir = '/ssd/leftImg8bit_sequence/train'

path = []
sequence = []
for folder in os.listdir(data_dir):
	cur = data_dir+'/'+folder
	if os.path.isdir(cur):
		seqs = set()
		for image in os.listdir(cur):
			seq = int(image[len(folder+'_'):len(folder+'_')+6])
			seqs.add(seq)
		for seq in seqs:
			path.append(cur)
			sequence.append(seq)


df = pd.DataFrame({'path': path, 'sequence': sequence})
df.to_csv("video_train_filelist.csv", sep=',', index=False)