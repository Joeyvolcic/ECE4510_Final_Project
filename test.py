import cv2 as cv
import cv2
import numpy as np
import time
from objloader_simple import *
from ar_main import *

#Variables used for copmutation
MIN_MATCH_COUNT = 2
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
#I dont need this bounding box, we dont want to transform this we want a close up image of the object that takes up the whole screen

#input images
input_image = cv.imread('C:\\Users\\JoeyV\\OneDrive\\4510\\mac.jpg')
input_image = cv2.resize(input_image, (480, 480)) #resize image to 720x720
#input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) #converts image to grayscale
obj = OBJ('C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\fox.obj', swapyz=True)  
print(obj)

#create objects for ORB and BFMatcher
#creating 2 ORBS so we can tweak the parameters for each one to get the best results
orb_input = cv.ORB_create(nfeatures = 10000, edgeThreshold = 25)#cv.ORB_create(nfeatures = 100,  nlevels = 30, WTA_K = 2, edgeThreshold = 100, patchSize=50)
orb_dynamic = cv.ORB_create(nfeatures = 100, edgeThreshold = 25)#cv.ORB_create(nfeatures = 100,  nlevels = 30, WTA_K = 2, edgeThreshold = 100, patchSize=50)

#creates bfmatcher object, can switch to other bfmatchers if needed
bfm = cv.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#finds the keypoints and descriptors for the input image
input_image_kp, input_image_des = orb_input.detectAndCompute(input_image,None)
input_image_keypoints= cv2.drawKeypoints(input_image, input_image_kp, None, color=(0,255,0), flags=0)

#shows first image with keypoints and bounding box
cv2.imshow("capture",input_image_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()

#starts the video capture 
cam = cv2.VideoCapture(0) 
cv2.namedWindow("test")
img_counter = 0

while True: 
    ret, frame = cam.read()
    frame = frame[0:480, 80:560] #crop the image to 480x480 to match the input image
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts image to grayscale 
    if not ret:
        print("failed to grab frame")
        break
    
    #Finds the key points in the current frame and matches the descriptors
    frame_kp, frame_des = orb_dynamic.detectAndCompute(frame,None)
    matches = bfm.match(input_image_des,  frame_des)

    # Sort them in the order of their distance, the 10 best matches are the first 10 in the list
    matches = sorted(matches, key = lambda x:x.distance)
    goodMatches = matches[:10]

    #gets the points from the best matches
    src_pts = np.float32([ input_image_kp[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    dst_pts = np.float32([ frame_kp[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

    # # Calculate the homography
    transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w = 480,480
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    
    
    # obtain 3D projection matrix from homography matrix and camera parameters
    perspective_transform = cv.perspectiveTransform(pts, transform)
    perspective_transform += (w, 0)
    # project cube or model
    projection = projection_matrix(camera_parameters, transform) 
    print(projection) 
    new = render(frame, obj, projection, input_image, False)
    #frame = render(frame, model, projection)
   

    #this function likes to break, so if it breaks we just skip this frame and use the pervious perspective transform we can speed the program up
    try:
        perspective_transform = cv.perspectiveTransform(pts, transform)
        perspective_transform += (w, 0)
    except:
       continue

    # plots the transformed bounding box pts on the current frame
    # frame_boundingbox = cv2.polylines(frame, [np.int32((perspective_transform*0.8))+75], isClosed , color, thickness)

    #finds the 10 best matches between the images, and only draws the first 10 points and lines
    image_matches = cv.drawMatches(input_image, input_image_kp, new, frame_kp, goodMatches,  None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image_matches = cv2.polylines(image_matches, [np.int32(perspective_transform)], True, (0,0,255),3, cv2.LINE_AA)
    try:
        cv2.imshow("test", image_matches)
    except:
         cv2.imshow("test", image_matches)
 
    
    #This waits for the user to press space and then saves the image with the matches plotted between the two images
    #stops the video capture and closes the windows
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE pressed 
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, image_matches)
        print("{} written!".format(img_name))
        img_counter += 1
        break

cam.release()
cv2.destroyAllWindows()





