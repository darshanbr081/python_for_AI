"""Vectorized Normalization: Write a function to normalize a 2D NumPy array (Z-score) without using loops."""

import numpy as np

def normalize_array(arr):
    """
    Normalize a 2D Numpy array using Z-score normalization.

    prams:
    arr (numpy.ndarray): A 2D array to be normalized.
    returns:
    numpy.ndarray: A normalized 2D array.
    """
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    normalized_arr = (arr - mean)/std
    return normalized_arr



""" Graphical Representation: Create a function that takes a 2D NumPy array and visualizes it using a heatmap with Matplotlib. """
import matplotlib.pyplot as plt
def plot_heatmap(arr):
    """
    Visualize a 2D Numpy array using a heatmap.
    params:
    arr (numpy.ndarray): A 2D array to be visualized.
    """
    plt.imshow(arr, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('heatmap visualization for 2D array')
    plt.xlabel('columns')
    plt.ylabel('rows')
    plt.show()

    #Example usage:
if __name__ =="__main__":
    arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    normalized_arr = normalize_array(arr)
    print(normalized_arr)
    plot_heatmap(normalized_arr)