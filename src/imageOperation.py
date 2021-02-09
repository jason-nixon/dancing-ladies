import numpy as np
import cv2 as cv

class ImageOperations():

    def __init__(self, _image):
        self.image = _image
    
    def transformToHSH(self):

    def transformToRGB(self):

        aFrame = cv.cvtColor(aFrame, cv.COLOR_GRAY2RGB)

    def transformToGRAY(self):

        aFrame = cv.cvtColor(aFrame, cv.COLOR_GRAY2RGB)

    def doOpening(self, _nKernalDimRow, _nKernalDimCol):

        self.image = cv.morphologyEx(
            src = self.image, 
            op = cv.MORPH_CLOSE, 
            kernel = __createKernal(_nKernalDimRow = _nKernalDimRow, _nKernalDimCol = _nKernalDimCol))

    def doClosing(self, _nKernalDimRow, _nKernalDimCol):

        self.image = cv.morphologyEx(
            src = self.image, 
            op = cv.MORPH_CLOSE, 
            kernel = __createKernal(_nKernalDimRow = _nKernalDimRow, _nKernalDimCol = _nKernalDimCol))

    def __createKernal(self, _nKernalDimRow, _nKernalDimCol):
        """Creates a row-col kernal for """

        return np.ones((_nKernalDimRow, _nKernalDimCol),np.uint8))

    def getImage(self):
        return self.image

    def removeSmallBodies(self):


    def closeClosedBodis(self):

# do operations only on portions of the frame
