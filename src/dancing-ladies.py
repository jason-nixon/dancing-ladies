import numpy as np
import cv2 as cv
import argparse

# Specify the filepath of the video to read.  

video_file = 'C:\\repos\\dancing-ladies\\asset\\video\\ladies.avi'

# Generate the background subtraction object (KNN or MOG2).  
backSub = cv.createBackgroundSubtractorKNN(history = 2, dist2Threshold = 75, detectShadows = False)

# Start reading the target video.  
capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_file))

if not capture.isOpened:
    print('Error: Unable to open: ' + args.input)
    exit(0)

index = 0

while True:

    # Incremenet frame index.  
    index += 1

    # Print current frame index.  
    print(index)

    # Read frame from video.  
    _, frame = capture.read()

    # If no more frames, done.  
    if frame is None:
        print('Warning: Frame is empty.')
        break

    # Convert color space RGB -> Gray
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # # Apply opening morphological operation.  
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Convert color space Gray -> RGB
    # frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

    # Convert colorspace: RGB -> HSV
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    
    # Calculate the forground mask.  
    fgMask = backSub.apply(frame)


    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)

    # Show the foreground mask.  
    cv.imshow(winname = 'ForegroundMask', mat = frame)
