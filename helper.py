import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow import keras

def zero50(arr):
    # Determine the threshold value for the top 50% elements
    threshold = np.percentile(arr, 50)

    # Create a copy of the original array
    arr_copy = np.copy(arr)

    # Set elements greater than the threshold to zero
    arr_copy[arr_copy > threshold] = 0


    return arr_copy

'''
arr = np.array([100, 389, 324, 235356, 36, 2, 5, 2, 7, 9, 3, 6, 8, 4])
result = set_top_50_percent_to_zero(arr)
print(result)
'''




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


