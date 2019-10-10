"""
Prav - Finding similar images
     - Potential to product recommendations and substitutes to searching product 
   
"""
import sys, os
import pandas as pd
import numpy as np

from src.helper_utils import find_topk_unique, plot_query_answer

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors

"""
Load pre-trained imagenet resnet50 model
"""
model = ResNet50(weights='imagenet')

"""
Read images and convert them to feature vectors
"""
imgs, filename_heads, X = [], [], []

path = "input_images"
print("Reading images from '{}' directory...\n".format(path))
for f in os.listdir(path):

    # Process filename
    filename = os.path.splitext(f)  # filename in directory
    filename_full = os.path.join(path,f)  # full path filename
    head, ext = filename[0], filename[1]
    if ext.lower() not in [".jpg", ".jpeg"]:
        continue

    # Read image file
    img = image.load_img(filename_full, target_size=(224, 224))  # load
    imgs.append(np.array(img))  # image
    filename_heads.append(head)  # filename head

    # Pre-process for model input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img).flatten()
    X.append(features)

X = np.array(X)
imgs = np.array(imgs)
print("imgs.shape = {}".format(imgs.shape))
print("X_features.shape = {}\n".format(X.shape))

"""
Find NearestNeighbors images to each image
"""
n_neighbours = 5 + 1  # add 1 for self image similarity and exclude at finding unique searches

kNN = NearestNeighbors(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
kNN.fit(X)
   
"""
Plot recommendations for each image in database
"""
output_rec_dir = os.path.join("output", "recommendations")
if not os.path.exists(output_rec_dir):
    os.makedirs(output_rec_dir)
n_imgs = len(imgs)
ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]

AllsearchImageName = []
AllsimilarImageName = []

for ind_query in range(n_imgs):
    
    searchImageName = []
    similarImageName = []

    print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query+1, n_imgs, filename_heads[ind_query]))
    distances, indices = kNN.kneighbors(np.array([X[ind_query]]))
    distances = distances.flatten()
    indices = indices.flatten()
    indices, distances = find_topk_unique(indices, distances, n_neighbours)

    # Plot recommendations
    rec_filename = os.path.join(output_rec_dir, "{}_similarities.png".format(filename_heads[ind_query]))
    x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
    x_answer_plot = imgs[indices].reshape((-1, ypixels, xpixels, 3))
    plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    searchImage = filename_heads[ind_query]
    AllsearchImageName.append(searchImage)
    
    for imageIndice in range(n_neighbours):
        print(imageIndice)
        print(filename_heads[indices[0][0]])
        recommendationImage = filename_heads[indices[0][imageIndice]]
        similarImageName.append(recommendationImage)
        
        
    AllsimilarImageName.append(similarImageName)


similarImages_df = pd.DataFrame(AllsimilarImageName)
similarImages_df.columns = ['similarity01', 'similarity02', 'similarity03', 'similarity04', 'similarity05', 'similarity06']

searchImages_df = pd.DataFrame(AllsearchImageName)
searchImages_df.columns = ['searchImage']

# write this dataframe to database for later use
similarImages = pd.concat([searchImages_df,similarImages_df], axis=1)







