import os
import glob
import time
import itertools

# outils SKlearn
# http://scikit-learn.org/stable/ --> machine learning Python
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib


# outils graphique
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# outils équivalents Matlab
import numpy as np
from scipy.ndimage.measurements import label

# openCV
import cv2

# dans le bon repertoire
os.getcwd()
# /!\ à VIRER  /!\ à VIRER  /!\ à VIRER  /!\ à VIRER  /!\ à VIRER  /!\ à VIRER  /!\ à VIRER 
os.chdir('/home/andarta/Documents/Udacity/16_VehiculeDetection/P05_Vehicle_detection_VT02')
os.getcwd()


##############################################################################
#
#  APPRENTISSAGE VEHICULE
#
##############################################################################



#----------------------------------------------------------------------
# Paramètres
#

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


#----------------------------------------------------------------------
# Les fonctions
#
# ...............................................
# fonction qui redimensionne l'image
# applati l'image pour prendre tous les pixels comme entre
# features = 16*16*3 = 768 entrees
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size) # image en 32*32*3
    #plt.imshow(features)
    #plt.show()    
    pixels_features = features.ravel() #image en 3072
    #print(pixels_features.shape)
    return pixels_features

# ...............................................
# fonction qui fait une histogramme sur les 3 couleurs
# domaine de l'étude -> 0 à 256
# 32 ou 16 raies
# features = 16*3 = 48 entrees
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # histo pour chaque couleur
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    #plt.hist(channel1_hist)
    #plt.show()  
    # Merge tous les histo des 3 couleurs
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    #print(hist_features.shape)
    #print(channel1_hist)
    return hist_features

# ...............................................
# fonction du filtre de HOG
# calcul du gradiant dans une petite case --> (pix_per_cell, pix_per_cell) --> cellule
# filtre avec les celluele voisines pour smoother --> (cell_per_block, cell_per_block) --> block
# http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
# transform_sqrt=True --> Apply power law compression to normalize the image before processing
# block_norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
# features =  1764 entrees
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
        #print(img.shape)
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=feature_vec)
        #plt.imshow(hog_image)
        #plt.show() 
        #print(features.shape)
        return features

    
# ...........................
# Fonction Principale d'extractions des caracteristiques des images
# Features 1 : tous les pixels image en 16*16*3
# Features 2 : histogramme 
# Features 3 : filtre de hog --> gradients
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    features = [] # caractéristiques de toutes les images pour apprentissage
    for file in imgs:
        file_features = []
        # Lecture une image après l'autres
        image = mpimg.imread(file)
        image = (image*255).astype(np.uint8)
        # passage dans un autre referentiel de couleur que RGB
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        # features 1 : tous les pixels sur image réduite
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        # features 2 : sommes des histo des 3 couleurs    
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # feature 3 : filtre de hog   
        if hog_feat == True:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        # toutes les features    
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# test extraction features
#ImagesTests = glob.glob('DataUdacity/Test/*.png')
#solution = extract_features(ImagesTests, color_space=color_space, 
#                        spatial_size=spatial_size, hist_bins=hist_bins, 
#                        orient=orient, pix_per_cell=pix_per_cell, 
#                        cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                        hist_feat=hist_feat, hog_feat=hog_feat)
#X = np.vstack((solution)).astype(np.float64)
#X.shape


#----------------------------------------------------------------------
# Extraction caracteristiques --> features
#
tdeb=time.time()
cars = glob.glob('DataUdacity/vehicles/**/*.png')
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcars = glob.glob('DataUdacity/non-vehicles/**/*.png')
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
tfin = time.time()
print(round(tfin-tdeb, 2), 'Seconds to extract features...')


#----------------------------------------------------------------------
# Prepare l'apprentissage
#
# les entrées
X = np.vstack((car_features, notcar_features)).astype(np.float64)  
# normalisation pour meilleur prise en compte
X_scaler = StandardScaler().fit(X) 
scaled_X = X_scaler.transform(X)                     
# les sorties à classifier
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# le partage base de données pour apprentissage / test
# attention --> X est normalisé !
np.random.seed(0xdeadbeef)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=0xdeadbeef)


#----------------------------------------------------------------------
# Apprentissage
#
# on utilise des SVM pour classifier nos images
# type de classificateur : Linear Support Vector Classification
# dans SKlearn : http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# option : squared_hinge ou hinge
clf = Pipeline([('scaling', StandardScaler()),
                ('classification', LinearSVC(loss='squared_hinge')),
               ])
#................................
# apprentissage
tdeb=time.time()
clf.fit(X_train, y_train)
tfin = time.time()
print(round(tfin-tdeb, 2), 'Seconds to train SVC...')

#----------------------------------------------------------------------
# Validation
#
print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))
tdeb=time.time()
n_predict = 25
print('Label real : ', y_test[0:n_predict])
print('Prediction : ', clf.predict(X_test[0:n_predict]))
tfin = time.time()
print(round(tdeb-tfin, 5), 'Seconds to predict', n_predict,'labels')

from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X)
conf = confusion_matrix(y, y_pred)
print(conf)










##############################################################################
#
#  Detection vehicule
#
##############################################################################

#................................
# idem apprentissage
# pour extraire les caractéristiques des sous-image --> sliding windows
# recopié d'Udacity
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


#................................
# recherche les sous-images dans une grande image
# en entrée --> la liste des potentielles sliding windows
# recopié d'Udacity
#
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        DesFunc = clf.decision_function(test_features)
        
        #print(DesFunc)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            if DesFunc > 0.5:
                on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

#................................
# definit les cadres des images glissantes
# qui vont etre données à analyser à search_windows
# search_windows va appliquer à chaque windows la fonction single_img_features pour extraire les features
#
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


#................................
# dessine un cadre sur la voiture
#
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

#................................
# supperpose toutes les fenetres glissantes et additionne un pixels
# au final on a des points chaud --> heatmap
# plus de fois on detecte une voiture, plus on sommes de pixel --> valeur plus haute
#
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

#................................
# pour virer les pixel trop faibles
# si la sommes de heatmap est inf à un critère alors le pixel = 0
# on fait ressortir que les points qui sont detecté plein de fois
# --> évite les fausses detections
#
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#................................
# dessine un cadre sur la voiture
# avec le heatmap
#
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img    



##############################################################################
#
#  Test sur une image
#
##############################################################################

#................................
# test 1 : sliding windows
#
test_img = mpimg.imread('test_images/test4.jpg')

windows =  slide_window(test_img,
                        x_start_stop=[None, None],
                        y_start_stop=[400, 656], #tune the parameters
                        xy_window=(64, 64),
                        xy_overlap=(0.5, 0.5))

window_img = draw_boxes(test_img, windows)
plt.imshow(window_img);
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
plt.title('Sliding Windows Technique:', fontsize=15);
plt.savefig('output_images/sliding_windows.png', bbox_inches="tight")


#................................
# test 2 : recherche des voiture simple
#

for i in range(4,5):
    
    fname = 'test_images/test{}.jpg'.format(i)
    image = mpimg.imread(fname)
    draw_image = np.copy(image)
    
    #step1 : def sliding windows
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], #tune the parameters
                            xy_window=(128,128),
                            xy_overlap=(.75,.75))
    #window_img = draw_boxes(image, windows)
    #plt.imshow(window_img);

    #step 2 : recherche voiture
    car_windows = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)   
   
    window_img = draw_boxes(draw_image, car_windows) 
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,9))
    plt.tight_layout()
    ax1.imshow(draw_image)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(window_img)
    ax2.set_title('Cars found', fontsize=10)
    plt.savefig('output_images/windows.png', bbox_inches="tight")


#................................
# test 3 : recherche des voiture avec point chaud
#

for i in range(4,12):
    
    fname = 'test_images/test{}.jpg'.format(i)
    image = mpimg.imread(fname)
    draw_image = np.copy(image)
    
    #step1 : def sliding windows
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], #tune the parameters
                            xy_window=(128,128),
                            xy_overlap=(.75,.75))
    #window_img = draw_boxes(image, windows)
    #plt.imshow(window_img);

    #step 2 : recherche voiture
    car_windows = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)   
                    
    #step 3 : concatenation sous forme de points chauds  
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)   
    heatmap2 = np.zeros_like(image[:,:,0]).astype(np.float)         
    heatmap = add_heat(heatmap, car_windows)
    # Apply threshold to help remove false positives
    heatmap2 = apply_threshold(heatmap,0.5)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmap2, 0, 255)
    labels = label(heatmap2)

    draw_img_label = draw_labeled_bboxes(image, labels)
    
        
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,9))
    plt.tight_layout()
    ax1.imshow(draw_image)
    ax1.set_title('Image', fontsize=10)
    ax2.imshow(heatmap)
    ax2.set_title('Heatmap', fontsize=10)
    ax3.imshow(draw_img_label)
    ax3.set_title('Image with car', fontsize=10)
    plt.savefig('output_images/windows.png', bbox_inches="tight")


#................................
# test 4 : recherche des voiture avec point chaud
# plusieur passage de taille differnete

for i in range(4,13):
    
    fname = 'test_images/test{}.jpg'.format(i)
    image = mpimg.imread(fname)
    draw_image = np.copy(image)
    
    #etape 1 :  recherche avec fenetre large 128*128
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], #tune the parameters
                            xy_window=(128,128),
                            xy_overlap=(.75,.75))
    car_windows1 = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)   
                   
    #etape 2 :  recherche avec fenetre moyenne 64*64
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], #tune the parameters
                            xy_window=(64,64),
                            xy_overlap=(.75,.75))
    car_windows2 = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)     
    #etape 3 :  on cacatène toutes les fenètres detectées 
    bboxlist = []
    bboxlist.append(car_windows1)
    bboxlist.append(car_windows2)
    bboxlist = [item for sublist in bboxlist for item in sublist]
    
    #etape 4 : concatenation sous forme de points chauds  
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)   
    heatmap2 = np.zeros_like(image[:,:,0]).astype(np.float)         
    heatmap = add_heat(heatmap, bboxlist)
    #étape 5 : critère mini sur les pixels pour virer les fausses detections
    heatmap2 = apply_threshold(heatmap,3)  # à régler /!\  
    heatmap = np.clip(heatmap2, 0, 255)
    labels = label(heatmap2)

    draw_img_label = draw_labeled_bboxes(image, labels)
    
        
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,9))
    plt.tight_layout()
    ax1.imshow(draw_image)
    ax1.set_title('Image', fontsize=10)
    ax2.imshow(heatmap)
    ax2.set_title('Heatmap', fontsize=10)
    ax3.imshow(draw_img_label)
    ax3.set_title('Image with car', fontsize=10)
    plt.savefig('output_images/windows.png', bbox_inches="tight")

#plt.imshow(heatmap)
#plt.imshow(heatmap2)


##############################################################################
#
#  Pour video
#
##############################################################################

# pour la memoire des heatmap
from collections import deque
heatmapque = deque(maxlen = 10) # 10 dernières images
# à régler /!\


def Video_Pipeline(image):   
    #etape 1 :  recherche avec fenetre large 128*128
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], #tune the parameters
                            xy_window=(128,128),
                            xy_overlap=(.75,.75))
    car_windows1 = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)   
                   
    #etape 2 :  recherche avec fenetre moyenne 64*64
    windows =  slide_window(image,
                            x_start_stop=[None, None],
                            y_start_stop=[400, 656], 
                            xy_window=(64,64),
                            xy_overlap=(.75,.75))
    car_windows2 = search_windows(image, windows, clf, 
                            X_scaler,
                            color_space=color_space, 
                            spatial_size=spatial_size, 
                            hist_bins=hist_bins, 
                            orient=orient, 
                            pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, 
                            hog_feat=hog_feat)   
    
    #etape 3 :  on cacatène toutes les fenètres detectées 
    bboxlist = []
    bboxlist.append(car_windows1)
    bboxlist.append(car_windows2)
    bboxlist = [item for sublist in bboxlist for item in sublist]
    
    #etape 4 : concatenation sous forme de points chauds  
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)   
    heatmap2 = np.zeros_like(image[:,:,0]).astype(np.float)  
    heatmap3 = np.zeros_like(image[:,:,0]).astype(np.float)        
    heatmap = add_heat(heatmap, bboxlist) 
    heatmap2 = apply_threshold(heatmap,3) # à régler /!\ 1 image
    heatmap = np.clip(heatmap2, 0, 255)
    
    #etape 5 : mémoire
    heatmapque.append(heatmap) #ajoute par la droite de la queue
    heatmapcombined = sum(heatmapque)
    heatmap3 = apply_threshold(heatmapcombined,6) # à régler /!\ 10 images
    heatmap = np.clip(heatmap3, 0, 255)
    
    labels = label(heatmap3)

    new_image = draw_labeled_bboxes(np.copy(image), labels)
    
    return new_image
    
from moviepy.editor import VideoFileClip

test_output = "test_video_output_03.mp4"
clip = VideoFileClip("test_video.mp4")
test_clip = clip.fl_image(Video_Pipeline)
test_clip.write_videofile(test_output, audio=False)

test_output = "project_short_output_03.mp4"
clip = VideoFileClip("project_short.mp4")
test_clip = clip.fl_image(Video_Pipeline)
test_clip.write_videofile(test_output, audio=False)

test_output = "project_video_output_06.mp4"
clip = VideoFileClip("project_video.mp4")
test_clip = clip.fl_image(Video_Pipeline)
test_clip.write_videofile(test_output, audio=False)















