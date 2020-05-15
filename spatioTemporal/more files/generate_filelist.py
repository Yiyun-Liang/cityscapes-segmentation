import pandas as pd
import numpy as np
import os

split = 'val'
data_dir = '/media/lmy/ssd/leftImg8bit_sequence/' + split

path = []
sequence = []
start = []
end = []
current = []
annotated_frames = []
indices = {}


annotation_dir = '../unet/data/imgs/' + split

for folder in os.listdir(annotation_dir):
	cur = annotation_dir+'/'+folder
	if os.path.isdir(cur):
		for image in os.listdir(cur):
			if not image.startswith('.'):
				seq = int(image[len(folder+'_'):len(folder+'_')+6])
				frame_id = int(image[len(folder+'_')+6+1:len(folder+'_')+6+1+6])
				# print(folder, seq, frame_id)
				key = folder + '/' + str(seq)
				if key not in indices:
					indices[key] = [frame_id] 
				else:
					indices[key].append(frame_id)


for folder in os.listdir(data_dir):
	cur = data_dir+'/'+folder
	if os.path.isdir(cur):
		start_frame = 0
		end_frame = 0
		i = 0
		find_end = False
		first = True
		paths = os.listdir(cur)
		paths.sort()
		#print(cur)
		for image in paths:
			if not image.startswith('.'):
				#print(cur + '/' + image)
				seq = int(image[len(folder+'_'):len(folder+'_')+6])
				frame_id = int(image[len(folder+'_')+6+1:len(folder+'_')+6+1+6])

				if first:
					start_frame = frame_id
					first = False

				else:
					# Find whether next frame exists
					next_path = cur + '/' + image[:len(folder+'_')] + str(seq).zfill(6) + '_' + str(frame_id+1).zfill(6) + image[len(folder+'_')+6+1+6:]
					#print(next_path)
					if not os.path.exists(next_path):
						end_frame = frame_id
						find_end = True
						#print(end_frame)
				i += 1
				if find_end:
					key = folder + '/' + str(seq)
					annotated_id = 0
					for item in indices[key]:
						if item <= end_frame and item >= start_frame:
							annotated_id = item

					for _ in range(i):
						end.append(end_frame)
						annotated_frames.append(annotated_id)
					i = 0
					find_end = False
					first = True

				sequence.append(seq)
				current.append(frame_id)
				start.append(start_frame)
				path.append(cur)

		

df = pd.DataFrame({'path': path, 'sequence': sequence, 'current': current, 'start': start, 'end': end, 'annotation': annotated_frames})
df.to_csv("video_"+split+"_filelist.csv", sep=',', index=False)
