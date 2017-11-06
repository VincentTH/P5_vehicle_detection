# P5_vehicle_detection


[//]: # (Image References)

[image1]: Image_base.png  "Basic"
[image2]: Image_resize.png "Resize"
[image3]: Image_YUV.png "Color"
[image4]: Image_histo.png "histogram"
[image5]: Image_HOG.png "HOG"
[image6]: Image_windows_sliding.png "sliding Windows"
[image7]: image_detection_basique.png "Basic detection"
[image8]: Image_point_chaud.png "One heat map"
[image9]: Image_point_chaud_simple_result.png "Multiple heat map"

[video1]: project_video_output_06.mp4 "Video"



# vehicule learning
## feature extraction
we have 3 types of features
we chnage the color format RGB --> YUV 
better to extarct features
![alt text][image1]
--> I test all other color ref... but YUV is the best
![alt text][image3]
1. all pixels of reduce image = 16*16*3 = 768 inputs for SVM
input image are 32*32*3 pixel, we reduce it to 16*16 and we flatten it
![alt text][image2]
2. histogram of this image
for each color chanel Y / U / V
with 16 bims
= 48 inputs for SVM
![alt text][image4]
3. HOG
determine gradient of the image
with same parameter as in in Udacity lecon
= 1764 input for SVM
![alt text][image5]

we apply it to car and not_car image database

## Back box model lerning
we chose a linear SVC
first we normalize inputs 
I keep 20 % of image for test
I use basic fit option
when I test it on whole test image, my prediction is 98.4 %
I reuse this predictor for car detection in full image

# vehicule detection
I go step by step
1. test sliding windows
![alt text][image6]
2. basic vehicule detection
sliding
serach vehicule
draw box
![alt text][image7]
3. more complex vehciuel detection
sliding
serach vehicule
define heat map
define a threshold on heatmap
search labels
draw box
![alt text][image8]
4. final vehicle detection
sliding windows
serach vehicule with histo + hog filter --> with big windows = 128*128
serach vehicule with histo + hog filter --> with small windows = 64*64
sum of all box liste
define heat map
define a threshold on heatmap
search labels
draw box
![alt text][image9]

to achive this, I use udacity function
- feature extraction = same as SVM learning
- sliding windows definition
- calculation of heat point
- function for remove low level heat pixel --> threshold

I chek it on test image
and on video

## isues and next step
no link between 2 frame
must intriduce a kalman filter (UKF or EKF) to predict the vehicule position and speed for next frame --> very easy to do with Matlab... but with python... I can't !


## new
I add in my function : search_windows
prediction = clf.predict(test_features)
DesFunc = clf.decision_function(test_features)
        if prediction == 1:
            if DesFunc > 0.5:
                on_windows.append(window)
that permit to reduce false positive

and I add also the fantastic tool with memory
    #etape 5 : mémoire
    heatmapque.append(heatmap) #ajoute par la droite de la queue
    heatmapcombined = sum(heatmapque)
    heatmap3 = apply_threshold(heatmapcombined,6) # à régler /!\ 10 images
    heatmap = np.clip(heatmap3, 0, 255)
    
    labels = label(heatmap3)

    new_image = draw_labeled_bboxes(np.copy(image), labels)

that permit to merge 10 previous images
--> filter car detection

new output is : project_video_output_06.mp4



