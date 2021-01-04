import numpy as np
import cv2 as cv

# Specify the filepath of the video to read.  
video_file = 'C:\\repos\\dancing-ladies\\asset\\video\\ladies.avi'

# Generate the background subtraction object (KNN or MOG2).  
backSub = cv.createBackgroundSubtractorKNN(history = 2, dist2Threshold = 75, detectShadows = False)

# Start reading the target video.  
capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_file))

# Quit is video could not be opened.  
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
    _, Frame = capture.read()

    # If no more frames, done.  
    if Frame is None:
        print('Warning: Frame is empty.')
        break

    # Convert color space RGB -> Gray
    Frame = cv.cvtColor(Frame, cv.COLOR_RGB2GRAY)

    # Apply opening morphological operation.  
    Frame = cv.morphologyEx(Frame, cv.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Convert color space Gray -> RGB
    Frame = cv.cvtColor(Frame, cv.COLOR_GRAY2RGB)

    # Convert colorspace: RGB -> HSV
    Frame = cv.cvtColor(Frame, cv.COLOR_RGB2HSV)
    
    # Calculate the forground mask.  
    ForegroundMask = backSub.apply(Frame)

    # Show the foreground mask.  
    cv.imshow(winname = 'ForegroundMask', mat = ForegroundMask)

    # Paused imshow for X miliseconds, imshow doesn't work otherwise (same as matlab).  
    keyboard = cv.waitKey(30)