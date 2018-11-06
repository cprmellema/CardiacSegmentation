"""
This file contains utility funtions to be used in the Cardiac
Segmentation Hackathon Challenge at the UTSW Hack-Med event on
Nov 9-10, 2018. The contributors to this file include:
Cooper Mellema
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

    def fFetchRawDataFile(self, sPath):
        NIIFile = sitk.ReadImage(os.path.join(sPath))
        return NIIFile

    def fNIIFileToArray(self, NIIFile):
        aDerivedImg = sitk.Cast(NIIFile, sitk.sitkFloat32)
        aDerivedImg = sitk.GetArrayViewFromImage(aDerivedImg)
        return aDerivedImg

    def fFetchTrainingData(self, sNIIFileName):
        NIIFile = self.fFetchRawDataFile((os.path.join(self.TrainDataLocation, sNIIFileName)))
        aDerivedImg = self.NIIFileToArray(NIIFile)
        return aDerivedImg

    def fFetchTestData(self):
        pdTestData=pd.DataFrame
        return pdTestData

