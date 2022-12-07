import numpy as np
import os
from collections import defaultdict
import tqdm
from multiprocessing import Pool, Manager
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', type=str, default='anns/EPIC_100_train.pkl')
parser.add_argument('--path_to_frames', type=str, default='/raid/asap7772/epic100/frames')
parser.add_argument('--output_path', type=str, default='/raid/asap7772/epic100/output')
parser.add_argument('--num_workers', type=int, default=64)
parser.add_argument('--max_frames', type=int, default=-1)
args = parser.parse_args()

data = np.load(args.annotation_path, allow_pickle=True)

data['noun_verb'] = data[['noun', 'verb']].agg('-'.join, axis=1)
vc = data['noun_verb'].value_counts()
vals = vc.values

des_labels = vc[vc > 100].keys()
new_data = data[data['noun_verb'].isin(des_labels)]
sorted_data = new_data.sort_values(by=['noun_verb'])

row_num = 0
filter_tasks =  []
if len(filter_tasks) > 0:
    print('Filtering tasks', filter_tasks)
    sorted_data = sorted_data[sorted_data['noun_verb'].isin(filter_tasks)]
else:
    print('No filter tasks')
    
dict_traj = {k: Manager().list() for k in des_labels}

def process_row(i):
    ind = sorted_data.index[i]
    folder_path= os.path.join(args.path_to_frames, sorted_data['participant_id'][ind], 'rgb_frames', str(sorted_data['video_id'][ind]))
    frame_names = sorted(os.listdir(folder_path))
    filter_value = lambda x: sorted_data['start_frame'][ind] <= int(x.split('.')[0].split('_')[-1]) <= sorted_data['stop_frame'][ind]
    filtered_frame_names = list(filter(filter_value, frame_names))
    full_frame_names = [os.path.join(folder_path, frame_name) for frame_name in filtered_frame_names]
    
    aux_data = sorted_data.iloc[i]
    task = aux_data['noun_verb']
    
    dict_traj[task].append((full_frame_names, aux_data))


if args.max_frames > 0:
    num_rows = args.max_frames
else:
    num_rows = sorted_data.shape[0]

num_workers = args.num_workers
pool = Pool(num_workers)
tqdm.tqdm.write('Processing data')
tqdm.tqdm.write('Number of workers: {}'.format(num_workers))
tqdm.tqdm.write('Number of tasks: {}'.format(num_rows))
tqdm.tqdm.write('Number of tasks per worker: {}'.format(num_rows // num_workers))

for _ in tqdm.tqdm(pool.imap_unordered(process_row, range(num_rows)), total=num_rows):
    pass
pool.close()
pool.join()

for k in dict_traj.keys():
    dict_traj[k] = list(dict_traj[k])


for k in dict_traj.keys():
    print(k, len(dict_traj[k]))
    
np.save(f'{args.output_path}/epic100_train_dict.npy', dict_traj)
