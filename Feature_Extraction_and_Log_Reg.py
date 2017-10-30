import numpy as np
import csv
import mahotas
import cv2
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

## Loading data (My crappy computer had trouble loading the entire 
## dataset in one take, so I split it into 5 csv files of 10000 each)
#x1 = np.loadtxt("train_x-000.csv", delimiter=",")
#y1 = np.loadtxt("train_y-000.csv", delimiter=",")
#x1 = x1.reshape(-1,64,64)
#y1 = y1.reshape(-1,1)
#x2 = np.loadtxt("train_x-001.csv", delimiter=",")
#y2 = np.loadtxt("train_y-001.csv", delimiter=",")
#x2 = x2.reshape(-1,64,64)
#y2 = y2.reshape(-1,1)
#x3 = np.loadtxt("train_x-002.csv", delimiter=",")
#y3 = np.loadtxt("train_y-002.csv", delimiter=",")
#x3 = x3.reshape(-1,64,64)
#y3 = y3.reshape(-1,1)
#x4 = np.loadtxt("train_x-003.csv", delimiter=",")
#y4 = np.loadtxt("train_y-003.csv", delimiter=",")
#x4 = x4.reshape(-1,64,64)
#y4 = y4.reshape(-1,1)
#x5 = np.loadtxt("train_x-004.csv", delimiter=",")
#y5 = np.loadtxt("train_y-004.csv", delimiter=",")
#x5 = x5.reshape(-1,64,64)
#y5 = y5.reshape(-1,1)
#x = np.concatenate((x1,x2,x3,x4,x5))
#y = np.concatenate((y1,y2,y3,y4,y5))
#x_test = np.loadtxt("test_x.csv", delimiter=",")
#x_test = x_test.reshape(-1,64,64)
##

## Apply Gaussian blurring on the images and perform threshold binarization
#X = np.zeros(x.shape)
#for i,I in enumerate(x):
#    I_blur = cv2.GaussianBlur(I, (3,3), 0)
#    _, dst = cv2.threshold(np.uint8(I_blur),190,1,cv2.THRESH_BINARY)
#    X[i,:,:] = dst
#    
#X_test = np.zeros(x_test.shape)
#for i,I in enumerate(x_test):
#    I_blur = cv2.GaussianBlur(I, (3,3), 0)
#    _, dst = cv2.threshold(np.uint8(I_blur),190,1,cv2.THRESH_BINARY)
#    X_test[i,:,:] = dst
#
#Y_flat = np.ravel(y)
##

## Function that extracts bounding rectangles from image
def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.cv.BoxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    return img_crop
##

## Extract Zernike features of image
#zernike_features = []
#for im in X:
#    image = im.copy()
#    (cnts, _) = cv2.findContours(np.uint8(im).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#    zernike_feats = []
#    for cnt in cnts[0:4]:
#        rect = cv2.minAreaRect(cnt)
#        img = crop_minAreaRect(image, rect)
#        #cv2.imshow("img", img)
#        zval = mahotas.features.zernike_moments(img, 30)
#        zernike_feats.append(zval)
#    zernike_features.append(zernike_feats)
#
#zernike_features_norm = [] # Reason for this is certain images produced less
# #than 4 contours (Probably images that did not binarize well). The following 
# #just pads the array with zeros (There might be a better option here)
#for feat in zernike_features:
#    if len(feat) == 3:
#        feat.append(np.zeros(25))
#    elif len(feat) == 2:
#        feat.extend([np.zeros(25),np.zeros(25)])
#    elif len(feat) == 1:
#        feat.extend([np.zeros(25),np.zeros(25),np.zeros(25)])
#    z_array = np.array(feat)
#    z_1_D = z_array.reshape((4*25,1))
#    zernike_features_norm.append(z_1_D.flatten())
#
#zernike_features_test = []
#for im in X_test:
#    image = im.copy()
#    (cnts, _) = cv2.findContours(np.uint8(im).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#    zernike_feats = []
#    for cnt in cnts[0:4]:
#        rect = cv2.minAreaRect(cnt)
#        img = crop_minAreaRect(image, rect)
#        #cv2.imshow("img", img)
#        zval = mahotas.features.zernike_moments(img, 30)
#        zernike_feats.append(zval)
#    zernike_features_test.append(zernike_feats)
#
#zernike_features_norm_test = [] # Reason for this is certain images produced less
# #than 4 contours (Probably images that did not binarize well). The following 
# #just pads the array with zeros (There might be a better option here)
#for feat in zernike_features_test:
#    if len(feat) == 3:
#        feat.append(np.zeros(25))
#    elif len(feat) == 2:
#        feat.extend([np.zeros(25),np.zeros(25)])
#    elif len(feat) == 1:
#        feat.extend([np.zeros(25),np.zeros(25),np.zeros(25)])
#    z_array = np.array(feat)
#    z_1_D = z_array.reshape((4*25,1))
#    zernike_features_norm_test.append(z_1_D.flatten())
##

## Visualize contours on image
#image = X[0].copy()
#(cnts, _) = cv2.findContours(np.uint8(image).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#image_cont = X[0].copy() # Draw contours function is destructive so this copy is
## made to be used.
#cv2.drawContours(image_cont, cnts, -1, (255, 255, 255), 2)
#cv2.imshow("Img", image_cont)
##

## Visualize bounded rectangle on contour in image
#rect = cv2.minAreaRect(cnts[1])
#img = crop_minAreaRect(image, rect)
#cv2.imshow("img", img)
##

## Using the binary image as features
#features = []
#for I in X:
#    features.append(I.flatten())
#
#features_test = []
#for I in X_test:
#    features_test.append(I.flatten())
##

## Histogram of Oriented Gradients extraction (Only applied on training data)
#hog_features = []
#visuals = []
#for im in X:
#    hog_image, vis = hog(im, orientations=9, pixels_per_cell=(4, 4),
#                    cells_per_block=(1, 1), visualise=True)
#    hog_features.append(hog_image)
#    visuals.append(vis)
##

## Zernike features extraction from image
#zernike_features = []
#for im in X:
#    zval = mahotas.features.zernike_moments(im, 30)
#    zernike_features.append(zval)
#
#zernike_features_test = []
#for im in X_test:
#    zval = mahotas.features.zernike_moments(im, 30)
#    zernike_features_test.append(zval)
##

## Hu moments extraction from image (Did not produce good results)
#hu_moments_features = []
#for im in X:
#    Mom = cv2.moments(im, binaryImage=True)
#    hu = cv2.HuMoments(Mom)
#    hu_moments_features.append(hu.flatten())
#    
#hu_moments_features_test = []
#for im in X_test:
#    Mom = cv2.moments(im, binaryImage=True)
#    hu = cv2.HuMoments(Mom)
#    hu_moments_features_test.append(hu.flatten())
##

## Training and Splitting the training data after feature extraction
#trainRI, testRI, trainRL, testRL = train_test_split(zernike_features_norm, Y_flat,
#                                                    test_size=0.25,random_state=40)

## K-Nearest Neighbor Model
#model = KNeighborsClassifier(n_neighbors=6)
#model.fit(trainRI, trainRL)
#acc = model.score(testRI, testRL)
#print("Validation accuracy: {:.2f}%".format(acc * 100))
##

## Logistic Regression Model
#logreg = linear_model.LogisticRegression()
#logreg.fit(trainRI, trainRL)
#acc = logreg.score(testRI, testRL)
#print("Validation accuracy: {:.2f}%".format(acc * 100))
##

## Applying the model of choice on the test data
#Pred_labels = logreg.predict(zernike_features_norm_test)
#Pred = Pred_labels.astype(int)
#with open("Output_4.csv", 'wb') as f:
#    writer = csv.writer(f, delimiter=",")
#    writer.writerow(list(Pred))