import numpy as np
import cv2 as cv
import os
from pathlib import Path
import config

DisplayFrames = False

GenerateOutputVideo = True

# Specify the filepath of the video to read.
video_file = 'C:\\repos\\dancing-ladies\\data\\raw\\ladies.avi'

# Generate the background subtraction object (KNN or MOG2).
backSub = cv.createBackgroundSubtractorKNN(history = 3, dist2Threshold = 75, detectShadows = False)

# Start reading the target video.
VideoInput = cv.VideoCapture(cv.samples.findFileOrKeep(video_file))

# Video input frame rate, width, and height (W and H must be cast as integers).
VideoInFrameRate = VideoInput.get(cv.CAP_PROP_FPS)
VideoinFrameWidth = int(VideoInput.get(cv.CAP_PROP_FRAME_WIDTH))
VideoinFrameHeight = int(VideoInput.get(cv.CAP_PROP_FRAME_HEIGHT))

# Quit is video could not be opened.
if not VideoInput.isOpened:
    print('Error: Unable to open: ' + args.input)
    exit(0)

if not os.path.isdir('C:\\repos\\dancing-ladies\\data\\processed\\'):
    os.mkdir('C:\\repos\\dancing-ladies\\data\\processed\\')

# Create video object to write to.
if GenerateOutputVideo:
    VideoOutput = cv.VideoWriter('C:\\repos\\dancing-ladies\\data\\processed\\output.avi', cv.VideoWriter_fourcc('M','J','P','G') , VideoInFrameRate, (VideoinFrameWidth,VideoinFrameHeight))

# Video frame index.
FrameIndex = 0

while True:

    # Incramenet and print frame index.
    FrameIndex += 1
    print(FrameIndex)

    # Read frame from video.
    _ret, Frame = VideoInput.read()

    # If no more frames, done.
    if Frame is None:
        print('--> Frame is empty, exiting while loop.')
        break

    # Convert color space RGB -> Gray
    Frame =  (cv.cvtColor(Frame, cv.COLOR_RGB2GRAY))

    # # Apply 'open' morphological operation to remove 'snow' (small unconnected bodies due to avi compression).
    # for index in range(0,3,1):
    #     Frame = cv.morphologyEx(Frame, cv.MORPH_OPEN, np.ones((index * 2 + 7, index * 2 + 7), np.uint8))

    
    # Convert color space Gray -> RGB
    Frame = cv.cvtColor(Frame, cv.COLOR_GRAY2RGB)

    # Convert colorspace: RGB -> HSV
    Frame = cv.cvtColor(Frame, cv.COLOR_RGB2HSV)

    # Calculate the forground mask.
    Frame = backSub.apply(Frame)
    
    # Find all connected entities.  
    nb_components, output, aStats, centroids = cv.connectedComponentsWithStats(Frame, connectivity = 8)
    
    # Remove the background (entry 0) from the connected component array.  
    aStats = aStats[1:, -1]

    # Minimum entitity size to keep.  
    nMinEntitySize = 25 

    # Generate a blank image to transplant to.  
    img2 = np.zeros((output.shape), dtype = np.uint8)

    # For every component in the image, keep if it's above minimum size.  
    for i in range(0,  nb_components - 1):
        if aStats[i] > nMinEntitySize:
            img2[output == i + 1] = 255

    Frame = img2

    Frame = cv.morphologyEx(src = Frame, op = cv.MORPH_CLOSE, kernel = np.ones((25, 3),np.uint8))

    Frame = cv.morphologyEx(src = Frame, op = cv.MORPH_CLOSE, kernel = np.ones((50, 5),np.uint8))

    # Frame = cv.morphologyEx(src = Frame, op = cv.MORPH_CLOSE, kernel = np.ones((100, 3),np.uint8))

    # Frame = cv.morphologyEx(src = Frame, op = cv.MORPH_CLOSE, kernel = np.ones((1, 200),np.uint8))

    contours = cv.findContours(Frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    for index in contours:
        cv.drawContours(Frame, [index], 0, (255,255,255), -1)

    # morph: open (erode, then dilate)
    # morph: close (dilate, then erode)
    # Morphological operations; manipulate kernal size
    # Close bodies.  
    


    # Frame = cv.morphologyEx(src = Frame, op = cv.MORPH_CLOSE, kernel = np.ones((300,1),np.uint8))

    # contours = cv.findContours(Frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # contours = contours[0] if len(contours) == 2 else contours[1]

    # for index in contours:
    #     cv.drawContours(Frame, [index], 0, (255,255,255), -1)


    # Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((100,5),np.uint8))

    # Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((1,50),np.uint8))



    # for index2 in range(0,2,1):

    #     for index in range(0,1,1):
    #         Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((7,7),np.uint8))

    #     for index in range(0,1,1):
    #         Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((1,100),np.uint8))

    # for index in range(0,1,1):
    #     Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((1,250),np.uint8))

    # Frame = cv.morphologyEx(Frame, cv.MORPH_CLOSE, np.ones((100,1),np.uint8))

    # segment bottom to waistline and do close operation (prevents arms from getting caught in the closing operation).  

    if DisplayFrames:
        # Show the foreground mask.
        cv.imshow(winname = 'ForegroundMask', mat = Frame)

        # Paused imshow for X miliseconds, imshow doesn't work otherwise (same as MATLAB).
        _ = cv.waitKey(30)

    # Write to output video object.
    if GenerateOutputVideo:
        VideoOutput.write(cv.cvtColor(Frame, cv.COLOR_GRAY2RGB))

# Close all open opencv windows.
cv.destroyAllWindows()

# Release output video.
if GenerateOutputVideo:
    VideoOutput.release()