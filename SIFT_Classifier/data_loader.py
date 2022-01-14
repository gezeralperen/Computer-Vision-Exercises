import cv2
import os
from random import shuffle
import numpy as np
    

class DataLoader():
    def __init__(self, train=True, do_shuffle=True) -> None:
        if train:
            self.path = 'Caltech20/training'
        else:
            self.path = 'Caltech20/testing'
        self.classes = [x for x in os.listdir(self.path) if x[0] != '.']
        
        self.data = []
        for cls in self.classes:
            files = os.listdir(f'{self.path}/{cls}')
            for file in files:
                    self.data.append((f'{self.path}/{cls}/{file}', cls))
        if do_shuffle:
            shuffle(self.data)
        self.class_dict = {}
        for i, cls in enumerate(self.classes):
            self.class_dict[cls] = i
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img = self.data[i][0]
        cls = self.data[i][1]
        
        img = cv2.imread(img)
        
        return (img, cls)