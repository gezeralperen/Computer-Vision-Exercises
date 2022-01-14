from data_loader import DataLoader
import numpy as np

from train import train
from datetime import datetime
import os


if __name__ == '__main__':
    
    os.system('clear')
    print(datetime.now().strftime("%H:%M:%S") + '\tSession started!')
    train_loader = DataLoader()
    for n_features in [2, 50, 100, 500]:
        print(f'##### RUNNING EXPERIMENT FOR K={n_features} #####')
        cluster, classifier, score, X_max, trials = train(train_loader,n_features=n_features)
        print(f'##### EXPERIMENT SCORE FOR K={n_features} : {score} #####')