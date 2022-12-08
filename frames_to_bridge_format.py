import numpy as np
import os
from collections import defaultdict
import tqdm
from multiprocessing import Pool, Manager
import argparse
from PIL import Image
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--npy_path', type=str, default='/raid/asap7772/epic100/output/epic100_train_dict.npy')
parser.add_argument('--path_to_frames', type=str, default='/raid/asap7772/epic100/frames')
parser.add_argument('--output_path', type=str, default='/raid/asap7772/epic100/epic100_bridgeform')
parser.add_argument('--domain_name', type=str, default='epic100')
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--split', type=float, default=0.9)
args = parser.parse_args()

if os.path.exists(args.output_path):
    print('Output path already exists. Removing.')
    os.system('rm -rf {}'.format(args.output_path))

desired_keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'aux_data']

data = np.load(args.npy_path, allow_pickle=True).item()
tasks = list(data.keys())

def make_dataset_summary(data):
    os.makedirs(args.output_path, exist_ok=True)
    file_path = os.path.join(args.output_path, 'dataset_summary.csv')
    tasks = list(data.keys())
    
    header = ['task number','entity', 'domain','task','number of demos']
    with open(file_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, task in enumerate(sorted(tasks)):
            row = [i, 'berkeley', args.domain_name, task, len(data[task])]
            writer.writerow(row)
make_dataset_summary(data)

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
    
    task_path = os.path.join(args.output_path, args.domain_name, task)
    
    folder_path = os.path.join(task_path, 'train')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    folder_path = os.path.join(task_path, 'val')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    for frame_paths, aux_data in tqdm.tqdm(data[task]):
    
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

    train_output_dict = {k: output_dict[k][:int(len(output_dict[k])*args.split)] for k in desired_keys}
    val_output_dict = {k: output_dict[k][int(len(output_dict[k])*args.split):] for k in desired_keys}
    train_rew = output_rew[:int(len(output_rew)*args.split)]
    val_rew = output_rew[int(len(output_rew)*args.split):]
    
    np.save(os.path.join(task_path, 'train', 'out.npy'), train_output_dict)
    np.save(os.path.join(task_path, 'val', 'out.npy'), val_output_dict)
    np.save(os.path.join(task_path, 'train', 'out_rew.npy'), train_rew)
    np.save(os.path.join(task_path, 'val', 'out_rew.npy'), val_rew)


print('Processing', len(tasks), 'tasks')

if args.parallel:
    with Pool(args.num_workers) as p:
        p.map(process_task, tasks)
else:
    for task in tasks:
        process_task(task)