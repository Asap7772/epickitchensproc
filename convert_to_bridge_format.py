import numpy as np
import os
from collections import defaultdict
import tqdm
from multiprocessing import Pool

annotation_path = 'anns/EPIC_100_train.pkl'
data = np.load(annotation_path, allow_pickle=True)

data['noun_verb'] = data[['noun', 'verb']].agg('-'.join, axis=1)
vc = data['noun_verb'].value_counts()
vals = vc.values

des_labels = vc[vc > 100].keys()
new_data = data[data['noun_verb'].isin(des_labels)]
sorted_data = new_data.sort_values(by=['noun_verb'])

row_num = 0
dict_traj = defaultdict(list)
path_to_frames='/raid/asap7772/epic100/frames'

filter_tasks =  []
if len(filter_tasks) > 0:
    print('Filtering tasks', filter_tasks)
    sorted_data = sorted_data[sorted_data['noun_verb'].isin(filter_tasks)]
else:
    print('No filter tasks')
    
    
for i in tqdm.tqdm(range(len(sorted_data))):
    ind = sorted_data.index[i]
    folder_path= os.path.join(path_to_frames, sorted_data['participant_id'][ind], 'rgb_frames', str(sorted_data['video_id'][ind]))
    frame_names = sorted(os.listdir(folder_path))
    filter_value = lambda x: sorted_data['start_frame'][ind] <= int(x.split('.')[0].split('_')[-1]) <= sorted_data['stop_frame'][ind]
    filtered_frame_names = list(filter(filter_value, frame_names))
    full_frame_names = [os.path.join(folder_path, frame_name) for frame_name in filtered_frame_names]
    
    assert len(filtered_frame_names) == sorted_data['stop_frame'][ind] - sorted_data['start_frame'][ind] + 1, 'Number of frames in the folder does not match the number of frames in the annotation'
    assert os.path.exists(folder_path) and all(map(os.path.exists, full_frame_names)), f'Folder {folder_path} does not exist'
    
    aux_data = sorted_data.iloc[i]
    
    dict_traj[sorted_data['noun_verb'][ind]].append((full_frame_names, aux_data))
    
print('Number of tasks', len(dict_traj.keys()))
print('Number of videos', sum([len(dict_traj[key]) for key in dict_traj.keys()]))
print('Number of frames', sum([len(dict_traj[key][i][0]) for key in dict_traj.keys() for i in range(len(dict_traj[key]))]))

output_path='/raid/asap7772/epic100/output'
np.save(f'{output_path}/epic100_train_dict.npy', dict_traj)
