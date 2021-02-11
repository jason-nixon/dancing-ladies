import numpy as np
import cv2 as cv
import os
from pathlib import Path
import config
import math

bDisplayFrames = True

bGenerateOutputVideo = False

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
# oBackgroundSubtractor = cv.createBackgroundSubtractorKNN(history = 3, dist2Threshold = 75, detectShadows = False)
# oBackgroundSubtractor = cv.createBackgroundSubtractorMOG2(history = 3, varThreshold = 7, detectShadows = False)
# oBackgroundSubtractor = cv.bgsegm.createBackgroundSubtractorCNT(
#     minPixelStability = 3,
#     useHistory = False,
#     maxPixelStability = 15,
#     isParallel = False
# )
oBackgroundSubtractor = cv.bgsegm.createBackgroundSubtractorGMG()


# Start reading the target video.
oInputVideo = cv.VideoCapture(cv.samples.findFileOrKeep(sVideoFullFilePath))

# Video input aWorkingFrame rate, width, and height (W and H must be cast as integers).
fFrameRate = oInputVideo.get(cv.CAP_PROP_FPS)
nFrameWidth = int(oInputVideo.get(cv.CAP_PROP_FRAME_WIDTH))
fFrameHeight = int(oInputVideo.get(cv.CAP_PROP_FRAME_HEIGHT))

# Quit if video could not be opened.
if not oInputVideo.isOpened:
    print('Error: Unable to open video.')
    exit(0)

if bGenerateOutputVideo: 
    if not os.path.isdir('C:\\repos\\dancing-ladies\\data\\processed\\'):
        os.mkdir('C:\\repos\\dancing-ladies\\data\\processed\\')

    if bGenerateOutputVideo:
        oOutputVideo = cv.VideoWriter('C:\\repos\\dancing-ladies\\data\\processed\\output.avi', cv.VideoWriter_fourcc('M','J','P','G') , fFrameRate, (nFrameWidth,fFrameHeight))

# Initialize aWorkingFrame index var.
nFrameIndex = 0

while True:

    # Increment and print frame index.
    nFrameIndex += 1
    print(nFrameIndex)

    # Read aWorkingFrame from video.
    bGrabbed, aFrameOriginal = oInputVideo.read()

    aFrameOriginal = aFrameOriginal[1:fFrameHeight, 641:1280, :]

    aWorkingFrame = np.copy(aFrameOriginal)

    aFrameOverlay = np.copy(aFrameOriginal)

    # If no more frames, done.
    if aWorkingFrame is None or aFrameOriginal is None:
        print('--> aWorkingFrame is empty, exiting while loop.')
        break

    # Convert color space RGB -> Gray
    # aWorkingFrame =  (cv.cvtColor(aWorkingFrame, cv.COLOR_RGB2GRAY))

    for index in range(0,1,1):
        aWorkingFrame = cv.morphologyEx(aWorkingFrame, cv.MORPH_OPEN, np.ones((index * 2 + 7, index * 2 + 7), np.uint8))

    aWorkingFrame = cv.medianBlur(aWorkingFrame, 5)

    aWorkingFrame = cv.bilateralFilter(aWorkingFrame, 15, 150, 150)

    aWorkingFrame = cv.cvtColor(aWorkingFrame, cv.COLOR_RGB2HSV)

    aMask = oBackgroundSubtractor.apply(aWorkingFrame)

    aEntityList, aLabeledImage, aEntityStats, aEntityCentroids = cv.connectedComponentsWithStats(image = aMask, connectivity = 8)

    aEntityStats = aEntityStats[1:, -1]

    aTempFrame = np.zeros((aLabeledImage.shape), dtype = np.uint8)    

    for nIndex in range(0,  aEntityList - 1):
        if aEntityStats[nIndex] > config.nMinEntitySize:
            aTempFrame[aLabeledImage == nIndex + 1] = 255

    aMask = aTempFrame

    aMask = cv.bilateralFilter(aMask, 7, 150, 150)

    aMask[np.where(aMask < 225)] = 0

    for nIndex in range(0, 20, 1):
        kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        aMask = cv.morphologyEx(src = aMask, op = cv.MORPH_CLOSE, kernel = kernal)

    for nIndex in range(0, 20, 1):
        aMask = cv.morphologyEx(src = aMask, op = cv.MORPH_OPEN, kernel = kernal)

    # for nIndex in range(0, 20, 1):
    #     aMask = cv.morphologyEx(src = aMask, op = cv.MORPH_CLOSE, kernel = kernal)

    # for nIndex in range(0, 20, 1):
    #     aMask = cv.morphologyEx(src = aMask, op = cv.MORPH_OPEN, kernel = kernal) 

    aFrameOverlay[np.where(aMask == 255)] = [0,255,255]

    cv.addWeighted(aFrameOverlay, 0.3, aFrameOriginal, 0.7, 0, aFrameOverlay)

    if bDisplayFrames and nFrameIndex % 2 == 0:

        # Show the foreground mask.
        cv.imshow(winname = 'ForegroundMask', mat = aFrameOverlay)

        # Paused imshow for X miliseconds, imshow doesn't work otherwise (same as MATLAB).
        _ = cv.waitKey(delay = 30)

    # Write to output video object.
    if bGenerateOutputVideo:
        oOutputVideo.write(aWorkingFrame)

    

# Close all open opencv windows.
cv.destroyAllWindows()

# Release output video.
if bGenerateOutputVideo:
    oOutputVideo.release()