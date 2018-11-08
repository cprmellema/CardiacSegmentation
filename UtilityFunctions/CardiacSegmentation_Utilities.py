"""
This file contains utility funtions to be used in the Cardiac
Segmentation Hackathon Challenge at the UTSW Hack-Med event on
Nov 9-10, 2018. The contributors to this file include:
Cooper Mellema
Paul Acosta
"""

import numpy as np
from keras import backend as K
import pandas as pd
import os
import sys
import SimpleITK as sitk


def fLogDiceloss(aPredictedVolumes, aActualVolumes):
    """
    This function returns the log of the dice score between two sets
    This function was originally implemented as a means of determining
    the accuracy of an automated segmentation deep learning algorithm
    in correctly classifying 3d Volumes. The Dice score is defined as:

            2|X Intersect Y|
    ______________________________
    |Cardinality X|+|Cardinality Y|    where X and Y are the labelled volumes

    :param aPredictedVolumes: a 3d array of the predicted volume labels
    :param aActualVolumes: a 3d array with the true volume labels
    :return: the log of the dice score
    """
    # May or may not need the tensor->array conversion, depending on how
    # output is formatted. No tests done on grayed out code
    # aPredictedVolumes = K.clip(aPredictedVolumes, K.epsilon(), 1-K.epsilon())

    flDiceLoss = ((2 * np.abs(np.intersect1d(aPredictedVolumes, aActualVolumes).size))
                   /(aPredictedVolumes.size + aActualVolumes.size)
                  )

    flLogDiceLoss = np.log(flDiceLoss)

    return flLogDiceLoss

class cPreprocess(object):
    """ This class contains all the methods for preprocessing the Cardiac Segmentation Data
    The general function types contained herein are as follows:
    -Functions to fetch Raw Data
    -Functions to fetch Training/Test Data
    -Functions to normalize Training and Test Data
    -Functions to reformat Training and Test Data
    """

    def __init__(self):
        self.RawDataLocation = '/project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation'
        self.ProcessedDataLocation = '/project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation'
        self.TrainDataLocation = os.path.join(self.RawDataLocation, 'mr_train')
        self.TestDataLocation = os.path.join(self.ProcessedDataLocation, 'mr_test')
        self.Dimension = 3 # 3 dimensional image

        # Initialize the reference image that other images will be resampled based on
        self.ReferenceImageParams = {'origin': np.zeros(self.Dimension), # sets the origin of the image to [0,0,0]
                                     'direction': np.identity(self.Dimension).flatten(), # sets direction to [1,1,1] (arbitrary)
                                     'size': [128]*self.Dimension, # downsample and/or upsample to 128x128x128
                                     'spacing': np.ones(self.Dimension) # set spacing to 1mm (arbitrary selection)
                                    }
        self.ReferenceImage = sitk.Image(self.ReferenceImageParams['size'], 2)
        self.ReferenceImage.SetOrigin(self.ReferenceImageParams['origin'])
        self.ReferenceImage.SetSpacing(self.ReferenceImageParams['spacing'])
        self.ReferenceImage.SetDirection(self.ReferenceImageParams['direction'])

    def fFetchRawDataFile(self, sPath):
        """Fetches a .nii file

        (currently redundant with sitk.ReadImage, but will be changed later)

        :param sPath: path to the .nii file
        :return: the .nii file in the sitk object form
        """
        NIIFile = sitk.ReadImage(sPath)
        return NIIFile

    def fResizeImage(self, NIIFile):
        """ Resizes a .nii file to parameters set in self

        :param NIIFile: .nii file to be set at origin, spacing, direction
        :return:
        """
        # cResampler=sitk.ResampleImageFilter()
        # cResampler.SetReferenceImage(self.ReferenceImage)
        # NIIResampled=cResampler.execute(NIIFile)
        NIIResampled=sitk.Resample(NIIFile, self.ReferenceImage)
        return NIIResampled
        
    def fPadImage(self, NIIFile):
        """
        Pads a .nii file to the parameters set in self

        :param NIIFile: .nii file to be set at origin, spacing, direction
        :return
        """
        NIIPadded = sitk.MirrorPad(NIIFile, self.ReferenceImage)
        return NIIPadded


    def fNIIFileToNormArray(self, NIIFile, flStd=1, flMean=0):
        """ Returns normalized array from the .nii file

        array is normalized by subtracting the mean and dividing by the std,
        if nothing is passed, just returns the array

        :param NIIFile: Take in a .nii file
        :return: returns an array from the .nii file
        """
        aDerivedImg = sitk.GetArrayFromImage(NIIFile)
        aDerivedImg = aDerivedImg - flMean
        aDerivedImg = np.divide(aDerivedImg, flStd)
        return aDerivedImg

    def fFetchTrainingData(self, sNIIFileName, **Args):
        """ Fetches Training data in raw form, normallizes it, and samples down to the same size
        Uses a pipeline of this form:
        Raw file -> fFetchRawDataFile -> imported raw nii file ->
        fResizeImage -> resize and Resampled nii file ->
        fNIIFileToNormArray -> normalized array

        :param sNIIFileName: the file name of the .nii file
        :return: array of normalized, resized data
        """
        NIIFile = self.fFetchRawDataFile((os.path.join(self.TrainDataLocation, sNIIFileName)))
        NIIFileResized = self.fResizeImage(NIIFile)
        aDerivedImg = self.fNIIFileToNormArray(NIIFileResized, **Args)
        return aDerivedImg

    def fFetchTestData(self):
        pdTestData=pd.DataFrame
        return pdTestData
    
    def fSave_ITK(self, sNIIFileName, sOutDir):
        NIIFile = self.fFetchRawDataFile((os.path.join(self.TrainDataLocation, sNIIFileName)))
        NIIFileResized = self.fResizeImage(NIIFile)
        sitk.WriteImage(NIIFileResized, os.path.join(sOutDir,sNIIFileName), True)

Preprocesser=cPreprocess()

# Convert all files to normalized arrays arrays after they have been preprocessed
for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
    Files.sort()
    aUnNormalizedAll = np.zeros(np.append(len(Files), Preprocesser.ReferenceImageParams['size']))
    for iFile, File in enumerate(Files):
        aUnNormalizedAll[iFile, :, :, :] = Preprocesser.fFetchTrainingData(File)

std = np.std(aUnNormalizedAll)
mean = np.mean(aUnNormalizedAll)

for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
    Files.sort()
    aNormalizedAll = np.zeros(np.append(len(Files), Preprocesser.ReferenceImageParams['size']))
    for iFile, File in enumerate(Files):
        aNormalizedAll[iFile, :, :, :] = Preprocesser.fFetchTrainingData(File, flStd=std, flMean=mean)

# Create a folder with resliced and resampled data saved as new .nii files
for Root, Dirs, Files in os.walk(Preprocesser.TrainDataLocation):
    Files.sort()
    for iFile, File in enumerate(Files):
        print(File)
        outDir= 'project/bioinformatics/DLLab/shared/Collab-Aashoo/WholeHeartSegmentation/mr_train_resized/'
        Preprocesser.fSave_ITK(File, outDir)

