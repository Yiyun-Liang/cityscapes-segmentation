import pandas as pd
import numpy as np
import os


data_dir = '/Volumes/stanford/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train'

path = []
sequence = []
start = []
end = []
for folder in os.listdir(data_dir):
	cur = data_dir+'/'+folder
	if os.path.isdir(cur):
		start_frame = 0
		end_frame = 0
		first = True
		last_seq = 0
		for image in os.listdir(cur):
			seq = int(image[len(folder+'_'):len(folder+'_')+6])
			frame_id = int(image[len(folder+'_')+6+1:len(folder+'_')+6+1+6])

			if first or frame_id != end_frame+1:
				if not first:
					path.append('/ssd/leftImg8bit_sequence/train/'+folder)
					sequence.append(last_seq)
					start.append(start_frame)
					end.append(end_frame)
				first = False
				start_frame = frame_id
				end_frame = frame_id
			else:
				end_frame = frame_id
			last_seq = seq
	

df = pd.DataFrame({'path': path, 'sequence': sequence, 'start': start, 'end': end})
df.to_csv("video_train_filelist.csv", sep=',', index=False)