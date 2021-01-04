import numpy as np
import cv2 as cv
from pathlib import Path

DisplayFrames = False

GenerateOutputVideo = True

# Specify the filepath of the video to read.  
video_file = 'C:\\repos\\dancing-ladies\\data\\raw\\ladies.avi'

# Generate the background subtraction object (KNN or MOG2).  
backSub = cv.createBackgroundSubtractorKNN(history = 2, dist2Threshold = 75, detectShadows = False)

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
    _, Frame = VideoInput.read()

    # If no more frames, done.  
    if Frame is None:
        print('Warning: Frame is empty, exiting while loop.')
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

    if DisplayFrames:
        # Show the foreground mask.  
        cv.imshow(winname = 'ForegroundMask', mat = ForegroundMask)

        # Paused imshow for X miliseconds, imshow doesn't work otherwise (same as MATLAB).  
        _ = cv.waitKey(30)

    # Write to output video object.
    if GenerateOutputVideo:
        VideoOutput.write(cv.cvtColor(ForegroundMask, cv.COLOR_GRAY2RGB))

# Close all open opencv windows.
cv.destroyAllWindows()

# Release output video. 
if GenerateOutputVideo:
    VideoOutput.release()