import os
import nilearn as nil
import numpy as np
import random
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image


class FmriDataset(Dataset):

    def __init__(self, params, data_dir='/data/fmri/data', mask_path='/data/fmri/mask/caudate_mask.nii',
                img_shape=(57, 68, 49, 135), img_timesteps=30):
        self.data_dir, self.params = data_dir, params
        self.img_timesteps = img_timesteps
        self.num_classes = params.nClass
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask_path, self.img_shape = mask_path, img_shape
        self.samples = []
        # Initialize the image indexes with their scores
        self.index_data()
        # self.mask = self.read_mask()
        self.class_weights = self.find_weights()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, score = self.samples[idx]
            score = self.get_class(score)
            img = self.read_image(img_path)
            # img = self.apply_mask(img)
            img = self.apply_temporal_aug(img)
            return img, score
        except Exception as error:
            print(error)
            print(self.samples[idx])
            with open('error.txt', 'a') as error_file:
                error_file.write(str(error) + '\n')
            return None

    def index_data(self):
        """
        Stores all the image_paths with their respective scores/classes in the 
        """
        self.weights = {i:0 for i in range(self.num_classes)}
        for sub in os.listdir(self.data_dir):
            if sub not in self.params.subs:
                continue
            sub_dir = os.path.join(self.data_dir, sub)
            preproc_dir = os.path.join(sub_dir, f'{sub}.preproc')
            for img_name in os.listdir(preproc_dir):
                img_path = os.path.join(preproc_dir, img_name)
                score = self.get_score(sub_dir, img_name)
                score_class = self.get_class(score)
                # Since we are randomly sampling 15 timesteps from each scan of 135 timesteps,
                # I am considering the same image for "n" times so that we have more data to train
                n = 5
                for k in range(n):
                    self.weights[score_class] += 1
                    self.samples.append((img_path, score))
                
    def get_class(self, score):
        """
        Categorize each score into one of the five classes (bins)
        Returns values from 0-4 (5 classes)
        """
#         if score < 1:
#             return 0
#         elif score >= 100:
#             return 4
#         else:
#             return score // 20
        
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
        

    def get_score(self, sub_dir, img_name):
        score_file = '0back_VAS-f.1D' if '0back' in img_name else '2back_VAS-f.1D'
        score_path = os.path.join(sub_dir, score_file)
        with open(score_path, 'r') as s_f:
            scores = [int(str(score.replace('\n', ''))) for score in s_f]

        task_num = img_name.split('.')[1]
        score_num = int(task_num[-1:])
        return scores[score_num]
    
    def read_image(self, img_path):
        nX, nY, nZ, nT = self.img_shape
        img = nil.image.load_img(img_path)
        img = img.get_fdata()[:nX, :nY, :nZ, :nT]
        img = torch.tensor(img, dtype=torch.float, device=self.device)
        img = (img - img.mean()) / img.std()
        return img
    
    def read_mask(self):
        nX, nY, nZ, _ = self.img_shape
        mask_img = nil.image.load_img(self.mask_path)
        mask_img = mask_img.get_fdata()[:]
        mask_img = np.asarray(mask_img)
        dilated_mask = np.zeros((nX, nY, nZ))
        ratio = round(mask_img.shape[2]/nZ)
        for k in range(nZ):
            temp = ndimage.morphology.binary_dilation(mask_img[:, :, k*ratio], iterations=1) * 1
            temp_img = Image.fromarray(np.uint8(temp*255))
            dilated_mask[:, :, k] = np.array(temp_img.resize((nY, nX)))
            
        dilated_mask = (dilated_mask > 64).astype(int)
        dilated_mask = torch.tensor(dilated_mask, dtype=torch.float, device=self.device)
        return dilated_mask
    
    def apply_mask(self, img):
        nT = img.shape[-1]
        for i in range(nT):
            img[:, :, :, i] = torch.mul(img[:, :, :, i], self.mask)
        return img
    
    def apply_temporal_aug(self, img):
        """
        Image shape: X, Y, Z, t=135
        So, e.g: take any 15 random timesteps from the 135 available in ascending order 
        """
        total_timesteps = img.shape[3]
        rand_timesteps = sorted(random.sample(range(0, total_timesteps), self.img_timesteps))
        # Move time axes to the first place followed by X, Y, Z
        img = img.permute(3, 0, 1, 2)
        img = torch.tensor(np.take(img.cpu().numpy(), rand_timesteps, axis=0))
        return img
    
    def find_weights(self):
        weights = dict(self.weights)
        key_max = max(weights.keys(), key=(lambda k: weights[k]))
        max_value = weights[key_max]
        for key in weights.keys():
            weights[key] = max_value / weights[key]
            
        return weights