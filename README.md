# P5_vehicle_detection

# vehicule learning
## feature extraction
we have 3 types of features
we chnage the color format RGB --> YUV 
better to extarct features
--> I test all other color ref... but YUV is the best
1. all pixels of reduce image = 16*16*3 = 768 inputs for SVM
input image are 32*32*3 pixel, we reduce it to 16*16 and we flatten it
2. histogram of this image
for each color chanel Y / U / V
with 16 bims
= 48 inputs for SVM
3. HOG
determine gradient of the image
with same parameter as in in Udacity lecon
= 1764 input for SVM

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
2. basic vehicule detection
sliding
serach vehicule
draw box
3. more complex vehciuel detection
sliding
serach vehicule
define heat map
define a threshold on heatmap
search labels
draw box
4. final vehicle detection
sliding windows
serach vehicule with histo + hog filter --> with big windows = 128*128
serach vehicule with histo + hog filter --> with small windows = 64*64
sum of all box liste
define heat map
define a threshold on heatmap
search labels
draw box

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









