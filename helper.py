import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tensorflow import keras
from matplotlib import pyplot as plt

plt.style.use('bmh')

def zero50(arr):
    threshold = np.percentile(arr, 50)
    arr_copy = np.copy(arr)
    arr_copy[arr_copy < threshold] = 0

    return arr_copy

def elim50(weightsR1, weightsR2):
    shape  = weightsR1.shape
    flat1 = weightsR1.flatten()
    flat2 = weightsR2.flatten()
    marks = np.zeros(len(flat1))
        
    marks = flat1 - flat2

    marks = zero50(marks)

    flat1[marks==0] = 0

    unflat1 = flat1.reshape(shape)

    return unflat1


def plotInstances(data, labels, clases):
    
    for i in range(10):
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 2))
        for j in range(5): 
            ax[j].imshow(data[labels.label.values == i,...][j])
            ax[j].grid()

        fig.tight_layout()

        fig.suptitle('Instacias de la clase ' + clases[i])
        plt.show()
    
def plotHist(data, labels, clases):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

    for i in range(5):
        ax[0, i].hist(data[labels.label.values == i,...].flatten(), 25, edgecolor='k')
        ax[0, i].set_title(clases[i])

    for i in range(5):
        ax[1, i].hist(data[labels.label.values == (i+5),...].flatten(), 25, edgecolor='k')
        ax[1, i].set_title(clases[i+5])

    fig.tight_layout()

    plt.show()
    
def plotAverages(data, labels, clases):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

    for i in range(5):
        ax[0, i].imshow(data[labels.label.values == i,...].mean(axis = 0))
        ax[0, i].set_title(clases[i])
        ax[0, i].grid()

    for i in range(5):
        ax[1, i].imshow(data[labels.label.values == (i+5),...].mean(axis = 0))
        ax[1, i].set_title(clases[i+5])
        ax[1, i].grid()
        
    fig.tight_layout()
    

    plt.show()
    
    
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

def displayAndSaveROC(trainY, testY, predict, path, name):
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()

    label_binarizer = LabelBinarizer().fit(trainY)
    y_onehot_test = label_binarizer.transform(testY)
    n_classes = 10

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    #print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        linestyle=':'
    )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    #plt.grid()


    plt.savefig(path + name +'.pdf')
    plt.show()


    
def plotAndSaveAcc(history, path, name):  
    plt.plot(history.history['val_accuracy'], 'o--', label='val_accuracy')
    plt.plot(history.history['accuracy'], 'o--', label='accuracy')
    plt.xticks
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    #plt.grid()
    plt.legend()
    plt.savefig(path + name +'acc.pdf')
    plt.show()
    
    
from contextlib import redirect_stdout
import json

def saveData(model, f1Macro, areaROC, history, path, name):
    with open(path + name + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
        f.write('\n')
        f.write('Loss: ' + model.loss)
        f.write('\n\n')
        f.write('Optimizer:\n')
        for key, value in model.optimizer.get_config().items(): 
            f.write('%s:%s\n' % (key, value))
        f.write('\n')
        f.write('f1 macro: ')
        f.write(str(f1Macro))
        f.write('\n')
        f.write('auc: ')
        f.write(str(areaROC))
        f.write('\n')
        f.write('acc: ')
        f.write(str(history.history['val_accuracy'][-1]))
        f.write('\n')
        f.write('loss: ')
        f.write(str(history.history['val_loss'][-1]))
        f.write('\n')
        f.write('precision: ')
        f.write(str(history.history['val_precision'][-1]))
        f.write('\n')
        f.write('recall: ')
        f.write(str(history.history['val_recall'][-1]))
        

def plotSeparate2D(outputOut, outputc2, combinations, labelsNp, clases):
    argm = np.zeros(len(outputOut))

    for idx, val in enumerate(outputOut):
        argm[idx] = int(np.argmax(val))

    for i in range(10):
        plt.scatter(combinations[argm==i, 0], combinations[argm==i, 1], alpha=0.2)

    for i in range(10):
        plt.scatter(outputc2[labelsNp==i, 0], outputc2[labelsNp==i, 1], alpha = 1, edgecolors='black', label=clases[i])

    plt.xlim(-30000, 30000)    
    plt.ylim(-30000, 30000) 
    plt.legend()

    plt.show()           
    
import os
    
def plotVarHyperAcc(): 
    cols = ['Val_accuracy', 'Accuracy', 'param']

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 5))

    for idx, file in enumerate(os.listdir('VarHyper')):
        if file.endswith('.txt'):
            arr = pd.read_csv('VarHyper/' + file, delimiter=',', names=cols)#, dtype=None, encoding='utf-8')

            (row, col) = (0, idx-1) if idx < 5 else (1, idx-5) 

            ax[row, col].plot(arr['param'], arr['Accuracy'], 'o--', label='accuracy')
            ax[row, col].plot(arr['param'], arr['Val_accuracy'], 'o--', label='val_accuracy')
            ax[row, col].set_xlabel(os.path.splitext(file)[0])
            ax[row, col].set_ylabel('Accuracy')
            if file == 'Batch size.txt':
                ax[row, col].set_xscale('symlog', base=2)   
            #ax[row, col].grid()
            ax[row, col].legend()


    fig.tight_layout()
    plt.show()
    
def plotVarHyperIter():
    cols = ['Val_accuracy', 'epochs', 'param']

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 5))

    for idx, file in enumerate(os.listdir('VarHyperIter')):
        if file.endswith('.txt'):
            arr = pd.read_csv('VarHyperIter/' + file, delimiter=',', names=cols)

            (row, col) = (0, idx-1) if idx < 5 else (1, idx-5) 
            if file == 'Batch size.txt':
                ax[row, col].plot(arr['param'], np.ceil(arr['epochs']*(45000/arr['param'])), 'o--', label='Iteraciones')
            else:
                ax[row, col].plot(arr['param'], np.ceil(arr['epochs']*(45000/256)), 'o--', label='Iteraciones')
            #ax[row, col].plot(arr['param'], arr['Val_accuracy'], 'o--', label='val_accuracy')
            ax[row, col].set_xlabel(os.path.splitext(file)[0])
            ax[row, col].set_ylabel('Iteraciones')
            if file == 'Batch size.txt':
                ax[row, col].set_xscale('symlog', base=2)   
            #ax[row, col].grid()
            ax[row, col].legend()


    fig.tight_layout()
    plt.show()
    
    
def plotLoss(history):
    plt.plot(history.history['loss'], 'o--', label='loss')
    plt.plot(history.history['val_loss'], 'o--', label='val_loss')
    #plt.ylim([0, 0.6])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.grid(True)
    plt.show()
    
    
def plotHistSlant(train_morpho, labels, clases):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

    for i in range(5):
        ax[0, i].hist(train_morpho['slant'][labels == i], 25, edgecolor='k')
        ax[0, i].set_title(clases[i])

    for i in range(5):
        ax[1, i].hist(train_morpho['slant'][labels == (i+5)], 25, edgecolor='k')
        ax[1, i].set_title(clases[i+5])

    fig.tight_layout()

    plt.show()
   
def plotScatters(train_morpho):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))

    for idx, (label, data) in enumerate(train_morpho.items()):
        if idx:
            (row, col) = (0, idx-1) if idx < 4 else (1, idx-4) 
            ax[row, col].scatter(train_morpho['slant'], data/max(data), alpha=0.01)
            ax[row, col].set_xlabel('Slant')
            ax[row, col].set_ylabel(label + ' (normalized)')

    fig.tight_layout()
    plt.show()