import os
import random

import numpy as np
import torch
import torchio as tio
from nilearn import image as nil_image
from scipy import ndimage
from PIL import Image
from torch.utils.data import Dataset

class FmriDataset(Dataset):
    
    def __init__(self, params, transform=None):
        self.params = params
        self.transform = transform
        
        self.samples = []
        # Initialize image paths (indexes) along with their fatigue levels (scores)
        self.index_data()
        self.class_weights = self.calculate_weights()
        if self.params.include_mask:
            self.mask = self.read_mask()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, score = self.samples[idx]
        score = self.get_class(score)
        img = self.read_image(img_path)
        if self.params.include_mask:
            img = self.apply_mask(img)
            
        # Apply temporal augmentation
        img = self.apply_temporal_aug(img)
        
        #return img, score
        return {"img": img, "label": score}
        
    def read_image(self, img_path):
        img = tio.ScalarImage(img_path).data
        # Make sure that the dimensions are uniform
        nX, nY, nZ = self.params.nX, self.params.nY, self.params.nZ
        img = img[:nX, :nY, :nZ, :]
        if self.transform:
            img = self.transform(img)
        
        img = img.float().to(self.params.device)
        return img
    
    def apply_temporal_aug(self, img):
        # Select random frames (n=seg_len) from all timesteps in ascending order
        total_timesteps = img.shape[3]
        rand_timesteps = sorted(random.sample(
            range(0, total_timesteps), self.params.seg_len))
        # Move time axes to the first place followed by X, Y, Z
        img = img.permute(3, 0, 1, 2)
        img = torch.tensor(np.take(img.cpu().numpy(), rand_timesteps, axis=0))
        return img
    
    def apply_mask(self, img):
        nT = img.shape[-1]
        for i in range(nT):
            img[:, :, :, i] = torch.mul(img[:, :, :, i], self.mask)
        return img
    
    def read_mask(self):
        mask_path = os.path.join(self.params.mask_dir, self.params.mask_type)
        nX, nY, nZ = self.params.nX, self.params.nY, self.params.nZ
        mask_img = nil_image.load_img(mask_path).get_fdata()[:]
        mask_img = np.asarray(mask_img)
        dilated_mask = np.zeros((nX, nY, nZ))
        ratio = round(mask_img.shape[2]/nZ)
        for k in range(nZ):
            temp = ndimage.morphology.binary_dilation(
                mask_img[:, :, k*ratio], iterations=1) * 1
            temp_img = Image.fromarray(np.uint8(temp*255))
            dilated_mask [:, :, k]= np.array(temp_img.resize((nY, nX)))
        dilated_mask = (dilated_mask > 64).astype(int)
        dilated_mask = torch.tensor(dilated_mask, dtype=torch.float, device=self.params.device)
        return dilated_mask
        
    def index_data(self):
        # For easy access of data through indices, store the paths in a list
        self.weights = {i: 0 for i in range(self.params.num_classes)}
        for sub in os.listdir(self.params.data_path):
            if sub not in self.params.current_subs:
                # Check if the sub belongs to the current set or not (works for both train and val sets))
                continue
            sub_dir = os.path.join(self.params.data_path, sub)
            preproc_dir = os.path.join(sub_dir, f'{sub}.preproc')
            for img_name in os.listdir(preproc_dir):
                img_path = os.path.join(preproc_dir, img_name)
                score = self.get_score(sub_dir, img_name)
                score_class = self.get_class(score)
                
                # Just an effort to increase the number of samples in the training set
                replicas = 5
                for _ in range(replicas):
                    self.weights[score_class] += 1
                    self.samples.append((img_path, score))
                    
    def get_score(self, sub_dir, img_name):
        score_file = '0back_VAS-f.1D' if '0back' in img_name else '2back_VAS-f.1D'
        score_path = os.path.join(sub_dir, score_file)
        with open(score_path, 'r') as s_f:
            scores = [int(str(score.replace('\n', ''))) for score in s_f]

        task_num = img_name.split('.')[1]
        score_num = int(task_num[-1:])
        return scores[score_num]
    
    def get_class(self, score):
        """
        Categorize each score into one of the five classes (bins)
        Returns values from 0-5 (6 classes)
        Classes: (0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)
        """
        if score < 10:
            return 0
        elif score >= 10 and score < 20:
            return 1
        elif score >= 20 and score < 40:
            return 2
        elif score >= 40 and score < 60:
            return 3
        elif score >= 60 and score < 80:
            return 4
        else:
            return 5
        
    def calculate_weights(self):
        weights = dict(self.weights)
        key_max = max(weights.keys(), key=(lambda k: weights[k]))
        max_value = weights[key_max]
        for key in weights.keys():
            # Add 1 to the denominator to avoid divide by zero error (in some cases)
            weights[key] = max_value / (weights[key]+1)

        return weights
                    
                    
        
