
import numpy as np
import matplotlib.pyplot as plt

"""
Disclaimer : copied these functions from open source code, not mine
"""

"""
 Finds the first unique k elements (based on lowest distance) of lists 'indices' and 'distances'
"""
def find_topk_unique(indices, distances, k):

    # Sort by ascending distance
    i_sort_1 = np.argsort(distances)
    distances_sorted = distances[i_sort_1]
    indices_sorted = indices[i_sort_1]

    window = np.array(indices_sorted[:k], dtype=int)  # collect first k elements for window intialization
    window_unique, j_window_unique = np.unique(window, return_index=True)  # find unique window values and indices
    j = k  # track add index when there are not enough unique values in the window
    # Run while loop until window_unique has k elements
    while len(window_unique) != k:
        # Append new index and value to the window
        j_window_unique = np.append(j_window_unique, [j])  # append new index
        window = np.append(window_unique, [indices_sorted[j]])  # append new value
        # Update the new unique window
        window_unique, j_window_unique_temp = np.unique(window, return_index=True)
        j_window_unique = j_window_unique[j_window_unique_temp]
        # Update add index
        j += 1

    # Sort the j_window_unique (not sorted) by distances and get corresponding
    # top-k unique indices and distances (based on smallest distances)
    distances_sorted_window = distances_sorted[j_window_unique]
    indices_sorted_window = indices_sorted[j_window_unique]
    u_sort = np.argsort(distances_sorted_window)  # sort

    distances_top_k_unique = distances_sorted_window[u_sort].reshape((1, -1))
    indices_top_k_unique = indices_sorted_window[u_sort].reshape((1, -1))

    return indices_top_k_unique, distances_top_k_unique


def plot_query_answer(x_query=None, x_answer=None, filename=None, gray_scale=False, n=5):

    # n = maximum number of answer images to provide
    plt.clf()
    plt.figure(figsize=(2*n, 4))

    # Plot query images
    for j, img in enumerate(x_query):
        if(j >= n):
            break
        ax = plt.subplot(2, n, j + 1)  # display original
        plt.imshow(img)
        if gray_scale:
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(4)  # increase border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title("searching Product",  fontsize=14)  # set subplot title

    # Plot answer images
    for j, img in enumerate(x_answer):
        if (j >= n):
            break

        ax = plt.subplot(2, n, n + j + 1)  # display original
        plt.imshow(img)
        if gray_scale:
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title("similar Product %d" % (j+1), fontsize=12)  # set subplot title

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')

    plt.close()
    
    
