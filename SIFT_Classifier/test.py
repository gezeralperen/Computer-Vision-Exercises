from datetime import datetime
from random import gammavariate
import cv2
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, plot_confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.metrics.pairwise import chi2_kernel

def custom_confusion(y_true, y_pred):
    size = np.max((np.max(y_true), np.max(y_pred))) + 1
    mat = np.zeros((size, size), dtype=np.int32)
    for i in range(y_true.shape[0]):
        true = y_true[i]
        pred = y_pred[i]
        mat[true, pred] += 1
    return mat

def test(test_loader, class_dict, cluster, classifier, X_max, testing_variables=True):
    classes_inverted = {v: k for k, v in class_dict.items()}
    classes_inverted = [classes_inverted[i] for i in range(len(classes_inverted))]
    
    sift = cv2.SIFT_create()
    embeddings = np.empty((0,128), dtype=np.float32)
    Y = np.empty((0,), dtype=np.int8)
    images = []
    indexes = []
        
    for i, (img, cls) in enumerate(test_loader):
        _, des = sift.detectAndCompute(img,None)
        if des is None:
            continue
        images.append((embeddings.shape[0], embeddings.shape[0]+des.shape[0]))
        embeddings = np.vstack((embeddings, des))
        Y = np.append(Y, class_dict[cls])
        indexes.append(i)
    
    print('Clustering is started!')
    
    clusters = cluster.predict(embeddings)
    
    # Calculate bag-of-feature-words
    X = np.zeros((len(images), cluster.n_clusters))
    for i, image in enumerate(images):
        feature_words = clusters[image[0]: image[1]]
        for j in range(feature_words.shape[0]):
            X[i, feature_words[j]] += 1
    
    
    X = X/X_max
    y_pred = classifier.predict(X)
    f1_per_class = f1_score(Y, y_pred, average=None)
    f1_macro = f1_score(Y, y_pred, average='macro')
    conf = custom_confusion(Y, y_pred)
    
    
    file = open(f'final_test/summary.txt', 'w+')
    
    #### Pretty Print ####    
    string = f'F1 Macro: {f1_macro}\n'
    string += f'F1 Per Class:\n'
    string += ' & '.join([f'{f1_per_class[i]:.2f}' for i in range(f1_per_class.shape[0])])
    string += '\nConfusion Matrix:\n'
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            string += f'{conf[i,j]:3d} & '
        string = string[:-2] + '\\\\\n'
    string += '\n---------------------------------------------------------------\n'
    print(string)
    file.write(string)
    file.close()
    
    df_cm = pd.DataFrame(conf, index = classes_inverted, columns = classes_inverted)
    plt.figure(figsize = (13,12))
    sn.heatmap(df_cm, annot=True, fmt='g', square=True)
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.title(f'Confusion Matrix on the Test Dataset')
    plt.savefig(f'final_test/cm-test.png', )
    
    
    wrong = [(test_loader[indexes[i]], Y[i], y_pred[i]) for i in range(Y.shape[0]) if Y[i] != y_pred[i]]
    
    for i in range(20):
        w = wrong[i]
        cv2.imwrite(f'false/{i}-{classes_inverted[w[1]]}-{classes_inverted[w[2]]}.jpg', w[0][0])
        