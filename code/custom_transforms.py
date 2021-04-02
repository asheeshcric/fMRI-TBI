import random

class Scale:
    
    def __init__(self, factor=(0.8, 1.0)):
        self.factor = factor
        
    def __call__(self, sample):
        # Each sample is a sequence of 3D-grayscale images: (85, 57, 68, 49)
        scale_factor = random.uniform(self.factor[0], self.factor[1])
        sample = sample * scale_factor
        return sample

        