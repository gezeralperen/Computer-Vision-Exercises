import cv2
from numpy.core.fromnumeric import shape
from data_loader import DataLoader
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.svm import SVC
import pickle
import os
from datetime import datetime
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def pretty_print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(f'{matrix[i,j]:3d}', end=' ')
        print('')

def evaluate_model(X, Y, classifier):
    k_fold = KFold(5, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, validation_ix in k_fold.split(X):
        train_x, train_y, validation_x, validation_y = X[train_ix], Y[train_ix], X[validation_ix], Y[validation_ix]

        classifier.fit(train_x, train_y)

        predicted_labels = classifier.predict(validation_x)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, validation_y)

    return predicted_targets, actual_targets


def train(train_loader, clustering = 'k-means', n_features = 100, classifier = 'SVM', training_variables=True, kernels=['linear', chi2_kernel], Cs=[1,10,20,30], path='summary'):
    
    classes_inverted = {v: k for k, v in train_loader.class_dict.items()}
    classes_inverted = [classes_inverted[i] for i in range(len(classes_inverted))]
    
    # Check if features are extracted before
    if training_variables and os.path.exists('training_variables.dat'):
        with open('training_variables.dat', 'rb+') as file:
            (embeddings, Y, images) = pickle.load(file)
    else:
        sift = cv2.SIFT_create()
        embeddings = np.empty((0,128), dtype=np.float32)
        Y = np.empty((0,), dtype=np.int8)
        images = []
        
        # Extract SIFT keypoints
        for (img, cls) in train_loader:
            _, des = sift.detectAndCompute(img,None)
            if des is None:
                continue
            images.append((embeddings.shape[0], embeddings.shape[0]+des.shape[0]))
            embeddings = np.vstack((embeddings, des))
            Y = np.append(Y, train_loader.class_dict[cls])
        with open('training_variables.dat', 'wb+') as file:
            pickle.dump((embeddings, Y, images),file)
            
    # Choose Clustering Algorithm
    if clustering == 'k-means':
        cluster = KMeans(n_features)
    if clustering == 'spectral':
        cluster = SpectralClustering(n_features)
    if clustering == 'agg':
        cluster = AgglomerativeClustering(n_features)
    
    # Calculate cluster centers
    clusters = cluster.fit_predict(embeddings)
    
    
    # Calculate bag-of-feature-words
    X = np.zeros((len(images), n_features))
    for i, image in enumerate(images):
        feature_words = clusters[image[0]: image[1]]
        for j in range(feature_words.shape[0]):
            X[i, feature_words[j]] += 1
    
    # Make classification
    X_max = np.max(X)
    X = X/X_max
    
    file = open(f'{path}/{n_features}.txt', 'w+')
    print('### GRID SEARCH STARTED ###')
    trials = []
    best_model = None
    best_score = 0
    for kernel in kernels:
        for C in Cs:
            clsfr = SVC(C=C, kernel=kernel, gamma=.5)
            y_pred, y_true = evaluate_model(X,Y,clsfr)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_per_class = f1_score(y_true, y_pred, average=None)
            conf = confusion_matrix(y_true, y_pred)
            if kernel == 'linear':
                kernel_string = 'linear'
            else:
                kernel_string = 'chi2'
            result = {
                'C' : C,
                'kernel' : kernel_string,
                'f1_macro' : f1_macro,
                'f1_per_class' : f1_per_class,
                'confusion_matrix' : conf
            }
            if f1_macro > best_score:
                best_model = clsfr
                best_score = f1_macro
            
            #### Pretty Print ####
            print(f'C = {C}\tKernel = {kernel_string}\tF1 Macro: {f1_macro}')
            print('F1 Per Class:')
            print(' '.join([f'{f1_per_class[i]:.2f}' for i in range(f1_per_class.shape[0])]))
            print('Confusion Matrix:')
            pretty_print_matrix(conf)
            print('---------------------------------------------------------')
            
            string = f'C = {C}\tKernel = {kernel_string}\tF1 Macro: {f1_macro}\n'
            string += f'F1 Per Class:\n'
            string += ' & '.join([f'{f1_per_class[i]:.2f}' for i in range(f1_per_class.shape[0])])
            string += '\nConfusion Matrix:\n'
            for i in range(conf.shape[0]):
                for j in range(conf.shape[1]):
                    string += f'{conf[i,j]:3d} & '
                string = string[:-2] + '\\\\\n'
            string += '\n---------------------------------------------------------------\n'
            file.write(string)
            trials.append(result)
            
            df_cm = pd.DataFrame(conf, index = classes_inverted, columns = classes_inverted)
            plt.figure(figsize = (13,12))
            sn.heatmap(df_cm, annot=True, fmt='g', square=True)
            plt.xlabel('Predicted Classes')
            plt.ylabel('True Classes')
            plt.title(f'Confusion Matrix for K={n_features} C={C} {kernel_string.capitalize()}')
            plt.savefig(f'{path}/cm-{n_features}-{kernel_string}-{C}.png', )
    file.close()
    return cluster, best_model, best_score, X_max, trials
    
