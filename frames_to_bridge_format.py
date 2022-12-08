import numpy as np
import os
from collections import defaultdict
import tqdm
from multiprocessing import Pool, Manager
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path', type=str, default='/raid/asap7772/epic100/output/epic100_train_dict.npy')
parser.add_argument('--path_to_frames', type=str, default='/raid/asap7772/epic100/frames')
parser.add_argument('--output_path', type=str, default='/raid/asap7772/epic100/epic100_bridgeform')
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--train', type=int, default=1)
args = parser.parse_args()

desired_keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'aux_data']

data = np.load(args.npy_path, allow_pickle=True).item()

tasks = list(data.keys())


def center_crop_pil(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

def resize_pil(img, new_width, new_height):
    return img.resize((new_width, new_height), Image.Resampling.BICUBIC)

def process_task(task):
    print('Processing task', task)
    # one task is a list of (frame_paths, aux_data)
    # create a numpy file for each task
    output_dict = {k: [] for k in desired_keys}
    output_rew = []
    sub_folder = 'train' if args.train else 'val'
    
    for frame_paths, aux_data in tqdm.tqdm(data[task]):
        folder_path = os.path.join(args.output_path, task, sub_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
    
        all_observations = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_aux_data = []
        # frame_path is a jpg file
        for frame_path in frame_paths:
            pil_image = Image.open(frame_path)
            pil_image = center_crop_pil(pil_image, 256, 256)
            pil_image = resize_pil(pil_image, 128, 128)
            
            frame = np.array(pil_image)
            assert frame.shape == (128, 128, 3)
            
            obs_dict = {
                'images0': frame,
                'images1': frame,
                'images2': frame,
                'state': np.zeros(7),
            }
            
            all_observations.append(obs_dict)
            all_actions.append(np.zeros(7))
            all_rewards.append(-1.0)
            all_terminals.append(0)
            all_aux_data.append(aux_data)
        
        observations, next_observations = all_observations[:-1], all_observations[1:]
        actions, rewards, terminals, aux_data = all_actions[:-1], all_rewards[:-1], all_terminals[:-1], all_aux_data[:-1]
        
        output_dict['observations'].append(observations)
        output_dict['next_observations'].append(next_observations)
        output_dict['actions'].append(actions)
        output_dict['rewards'].append(rewards)
        output_dict['terminals'].append(terminals)
        output_dict['aux_data'].append(aux_data)
        output_rew.append(rewards)

    output_path = os.path.join(args.output_path, task, sub_folder, 'out.npy')
    output_path_rew = os.path.join(args.output_path, task, sub_folder, 'out_rew.npy')
    
    np.save(output_path, output_dict)
    np.save(output_path_rew, output_rew)
    
    print('Saved to', output_path)

print('Processing', len(tasks), 'tasks')

if args.parallel:
    with Pool(args.num_workers) as p:
        p.map(process_task, tasks)
else:
    for task in tasks:
        process_task(task)