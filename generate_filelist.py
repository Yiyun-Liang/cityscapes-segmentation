import pandas as pd
import numpy as np
import os

split = 'test'
data_dir = '/Volumes/stanford/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/' + split

path = []
sequence = []
start = []
end = []
indices = {}
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
					path.append('/ssd/leftImg8bit_sequence/' + split + '/' +folder)
					sequence.append(last_seq)
					start.append(start_frame)
					end.append(end_frame)
					if folder+'_'+str(last_seq) not in indices:
						indices[folder+'_'+str(last_seq)] = []
					indices[folder+'_'+str(last_seq)].append(len(path)-1)
				first = False
				start_frame = frame_id
				end_frame = frame_id
			else:
				end_frame = frame_id
			last_seq = seq
		path.append('/ssd/leftImg8bit_sequence/' + split + '/' +folder)
		sequence.append(last_seq)
		start.append(start_frame)
		end.append(end_frame)
		if folder+'_'+str(last_seq) not in indices:
			indices[folder+'_'+str(last_seq)] = []
		indices[folder+'_'+str(last_seq)].append(len(path)-1)

data_dir = 'baseline/data/imgs/' + split

annotated_frames = [-1 for i in range(len(path))]
for folder in os.listdir(data_dir):
	cur = data_dir+'/'+folder
	if os.path.isdir(cur):
		for image in os.listdir(cur):
			seq = int(image[len(folder+'_'):len(folder+'_')+6])
			frame_id = int(image[len(folder+'_')+6+1:len(folder+'_')+6+1+6])
			# print(folder, seq, frame_id)
			cur = indices[folder+'_'+str(seq)]
			for c in cur:
				s = start[c]
				e = end[c]
				if frame_id <= e and frame_id >= s:
					annotated_frames[c] = frame_id

df = pd.DataFrame({'path': path, 'sequence': sequence, 'start': start, 'end': end, 'annotation': annotated_frames})
df.to_csv("video_"+split+"_filelist.csv", sep=',', index=False)