import numpy as np
import cv2 as cv
import os
from pathlib import Path
import config

bDisplayFrames = False

bGenerateOutputVideo = True

# Variable naming guide:  lower camel case (dromedary case).
# a_ : array
# b_ : boolean
# f_ : floating point number
# n_ : integer
# o_ : object
# s_ : string


# create a class that encapsulates and 'simply' parameterizs all the image processing
# steps that are currently being used.  The class should hold a local instance of the 
# frame, for speed.  

# Specify the filepath of the video to read.
sVideoFullFilePath = 'C:\\repos\\dancing-ladies\\data\\raw\\ladies.avi'

# Generate the background subtraction object (KNN or MOG2). (2, 75)
oBackgroundSubtractor = cv.createBackgroundSubtractorKNN(history = 2, dist2Threshold = 50, detectShadows = False)

# Start reading the target video.
oInputVideo = cv.VideoCapture(cv.samples.findFileOrKeep(sVideoFullFilePath))

# Video input aFrame rate, width, and height (W and H must be cast as integers).
fInputVideoFramerate = oInputVideo.get(cv.CAP_PROP_FPS)
nInputVideoFrameWidth = int(oInputVideo.get(cv.CAP_PROP_FRAME_WIDTH))
nInputVideoFrameHeight = int(oInputVideo.get(cv.CAP_PROP_FRAME_HEIGHT))

# Quit is video could not be opened.
if not oInputVideo.isOpened:
    print('Error: Unable to open video.')
    exit(0)

if not os.path.isdir('C:\\repos\\dancing-ladies\\data\\processed\\'):
    os.mkdir('C:\\repos\\dancing-ladies\\data\\processed\\')

# Create video object to write to.
if bGenerateOutputVideo:
    oOutputVideo = cv.VideoWriter('C:\\repos\\dancing-ladies\\data\\processed\\output.avi', cv.VideoWriter_fourcc('M','J','P','G') , fInputVideoFramerate, (nInputVideoFrameWidth,nInputVideoFrameHeight))

# Initialize aFrame index var.
nFrameIndex = 0



while True:

    # Incramenet and print aFrame index.
    nFrameIndex += 1
    print(nFrameIndex)

    # Read aFrame from video.
    bGrabbed, aFrameOriginal = oInputVideo.read()

    aFrame = aFrameOriginal

    # If no more frames, done.
    if aFrame is None or aFrameOriginal is None:
        print('--> aFrame is empty, exiting while loop.')
        break

    # Convert color space RGB -> Gray
    aFrame =  (cv.cvtColor(aFrame, cv.COLOR_RGB2GRAY))

    # # Apply 'open' morphological operation to remove 'snow' (small unconnected bodies due to avi compression).
    # for index in range(0,3,1):
    #     aFrame = cv.morphologyEx(aFrame, cv.MORPH_OPEN, np.ones((index * 2 + 7, index * 2 + 7), np.uint8))


    # Convert color space Gray -> RGB
    aFrame = cv.cvtColor(aFrame, cv.COLOR_GRAY2RGB)

    # Convert colorspace: RGB -> HSV
    aFrame = cv.cvtColor(aFrame, cv.COLOR_RGB2HSV)

    # Calculate the forground mask.
    aFrame = oBackgroundSubtractor.apply(aFrame)

    # Find all connected entities.
    aEntityList, aLabeledImage, aEntityStats, aEntityCentroids = cv.connectedComponentsWithStats(image = aFrame, connectivity = 8)

    # The background is included as the 0th entry of the entity statistics, and should be removed.
    aEntityStats = aEntityStats[1:, -1]

    # Generate a blank image to transplant to.
    aTempFrame = np.zeros((aLabeledImage.shape), dtype = np.uint8)

    # For every component in the image, keep if it's above minimum size.
    for i in range(0,  aEntityList - 1):
        if aEntityStats[i] > config.nMinEntitySize:
            aTempFrame[aLabeledImage == i + 1] = 255

    aFrame = aTempFrame

    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((25, 3),np.uint8))

    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE,kernel = np.ones((50, 5),np.uint8))

    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE,kernel = np.ones((3, 50),np.uint8))

    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((100, 3),np.uint8))

    contours = cv.findContours(aFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    for nIndex in contours:
        cv.drawContours(aFrame, [nIndex], 0, (255,255,255), -1)


    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((1, 200),np.uint8))

    aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((300, 1),np.uint8))

    # aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((100, 5),np.uint8))

    # aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((1, 200),np.uint8))

    # morph: open (erode, then dilate)
    # morph: close (dilate, then erode)
    # Morphological operations; manipulate kernal size
    # Close bodies.



    # aFrame = cv.morphologyEx(src = aFrame, op = cv.MORPH_CLOSE, kernel = np.ones((300,1),np.uint8))

    # contours = cv.findContours(aFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # contours = contours[0] if len(contours) == 2 else contours[1]

    # for index in contours:
    #     cv.drawContours(aFrame, [index], 0, (255,255,255), -1)


    # aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((100,5),np.uint8))

    # aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((1,50),np.uint8))



    # for index2 in range(0,2,1):

    #     for index in range(0,1,1):
    #         aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((7,7),np.uint8))

    #     for index in range(0,1,1):
    #         aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((1,100),np.uint8))

    # for index in range(0,1,1):
    #     aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((1,250),np.uint8))

    # aFrame = cv.morphologyEx(aFrame, cv.MORPH_CLOSE, np.ones((100,1),np.uint8))

    # segment bottom to waistline and do close operation (prevents arms from getting caught in the closing operation).

    aFrameOriginal[(aFrame>=1).all(-1)] = [0,255,0]


    if bDisplayFrames:
        # Show the foreground mask.
        cv.imshow(winname = 'ForegroundMask', mat = aFrame)

        # Paused imshow for X miliseconds, imshow doesn't work otherwise (same as MATLAB).
        _ = cv.waitKey(delay = 30)

    # Write to output video object.
    if bGenerateOutputVideo:
        oOutputVideo.write(aFrameOriginal)

# Close all open opencv windows.
cv.destroyAllWindows()

# Release output video.
if bGenerateOutputVideo:
    oOutputVideo.release()