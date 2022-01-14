from data_loader import DataLoader
import numpy as np

from train import train
from test import test
from sklearn.metrics.pairwise import chi2_kernel
from datetime import datetime
import os

n_features = 500
Cs = [30]
kernels = [chi2_kernel]

if __name__ == '__main__':
    
    os.system('clear')
    print(datetime.now().strftime("%H:%M:%S") + '\tSession started!')
    train_loader = DataLoader()
    print(f'##### Training with the Best Parameters #####')
    cluster, classifier, score, X_max, _ = train(train_loader,n_features=n_features, Cs=Cs, kernels=kernels, path='final_test')
    print(f'##### Training is completed with F1 macro score of  {score} #####')
    
    test_loader = DataLoader(train=False)
    test(test_loader, train_loader.class_dict, cluster, classifier, X_max)
    
    