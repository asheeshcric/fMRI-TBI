import os
import subprocess


"""
NOTE: Install "pigz" package for faster compression
"""

data_path = '/data/fmri/raw_data'


for sub_name in os.listdir(data_path):
    sub_dir = os.path.join(data_path, sub_name)
    for episode in os.listdir(sub_dir):
        episode_dir = os.path.join(sub_dir, episode)
        if not os.path.isdir(episode_dir):
            # Ignore text files
            continue
            
        # Also, check if the conversion has already been done or not
        file_names = os.listdir(episode_dir)
        if any([True for f_name in file_names if '.nii' in f_name]):
            continue
            
        command = f'dcm2niix -o {episode_dir}/ {episode_dir}/'
        p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        out, err = p.communicate()
    
    print(f'{sub_dir} done...')