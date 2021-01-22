import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk

import torch
import torch.utils.data as data

import getTrainingData as G

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims=True, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [2, 3], 'Supports only 2D (HxW) or 3D (CxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 2:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


def adjustData(img, mask):

    img = img.astype(np.float)/255.0

    mask = mask.astype(np.float)
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0

    return (img, mask)


class createDataset(data.Dataset):
    def __init__(self, allImageNames, allMaskImageNames, batch_size=100, patch_size=(64, 64)):
        self._batchSize = batch_size
        self._allImageNames = allImageNames
        self._allMaskImageNames = allMaskImageNames
        self._targetSize = patch_size

        self._n = 1
        self._trainLen = len(self._allImageNames)*self._batchSize*self._n

        n = len(self._allImageNames)
        print("Totally", n, "training images")

        self._imageAll = []
        self._maskAll = []

        for it in range(n):
            thisMaskName = self._allMaskImageNames[it]
            thisImageName = self._allImageNames[it]

            #thisImg = io.imread(thisImageName)
            #thisMask = io.imread(thisMaskName)

            thisImg = sitk.ReadImage(thisImageName)
            thisImg = sitk.GetArrayFromImage(thisImg)
            if thisImg.ndim == 3:
                thisImg = np.squeeze(thisImg, axis=0)

            thisMask = sitk.ReadImage(thisMaskName)
            thisMask = sitk.GetArrayFromImage(thisMask)
            if thisMask.ndim == 3:
                thisMask = np.squeeze(thisMask, axis=0)

            thisImg, thisMask = adjustData(thisImg, thisMask)

            self._imageAll.append(thisImg)
            self._maskAll.append(thisMask)
            
    def __getitem__(self, index):

        ImageIndex = index//self._batchSize

        thisImg = self._imageAll[ImageIndex]
        thisMask = self._maskAll[ImageIndex]

        sz = thisImg.shape
        patchShape = self._targetSize

        while 1:

            allPatchTopLeftX = np.random.randint(0, sz[0]-patchShape[0], size=1)
            allPatchTopLeftY = np.random.randint(0, sz[1]-patchShape[1], size=1)

            thisTopLeftX = allPatchTopLeftX[0]
            thisTopLeftY = allPatchTopLeftY[0]

            thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]

            if thisLabelImPatch.max() > 0:
                thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]

                T  = ToTensor()
                
                tensorImagePatch = T(thisImPatch)
                tensorLabelPatch = T(thisLabelImPatch)

                return tensorImagePatch, tensorLabelPatch

    def __len__(self):
        return self._trainLen

class withNegetiveSampleDataset(data.Dataset):
    def __init__(self, allImageNames, allMaskImageNames, batch_size=100, patch_size=(64, 64)):
        self._batchSize = batch_size
        self._allImageNames = allImageNames
        self._allMaskImageNames = allMaskImageNames
        self._targetSize = patch_size

        self._n = 1
        self._trainLen = len(self._allImageNames)*self._batchSize*self._n

        n = len(self._allImageNames)
        print("Totally", n, "training images")

        self._imageAll = []
        self._maskAll = []

        for it in range(n):
            thisMaskName = self._allMaskImageNames[it]
            thisImageName = self._allImageNames[it]

            #thisImg = io.imread(thisImageName)
            #thisMask = io.imread(thisMaskName)

            thisImg = sitk.ReadImage(thisImageName)
            thisImg = sitk.GetArrayFromImage(thisImg)
            if thisImg.ndim == 3:
                thisImg = np.squeeze(thisImg, axis=0)

            thisMask = sitk.ReadImage(thisMaskName)
            thisMask = sitk.GetArrayFromImage(thisMask)
            if thisMask.ndim == 3:
                thisMask = np.squeeze(thisMask, axis=0)

            thisImg, thisMask = adjustData(thisImg, thisMask)

            self._imageAll.append(thisImg)
            self._maskAll.append(thisMask)
            
    def __getitem__(self, index):

        ImageIndex = index//self._batchSize

        thisImg = self._imageAll[ImageIndex]
        thisMask = self._maskAll[ImageIndex]

        sz = thisImg.shape
        patchShape = self._targetSize

        allPatchTopLeftX = np.random.randint(0, sz[0]-patchShape[0], size=1)
        allPatchTopLeftY = np.random.randint(0, sz[1]-patchShape[1], size=1)

        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]

        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]

        thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]

        T  = ToTensor()
                
        tensorImagePatch = T(thisImPatch)
        tensorLabelPatch = T(thisLabelImPatch)

        return tensorImagePatch, tensorLabelPatch


    def __len__(self):
        return self._trainLen

class symmetryDataset(data.Dataset):
    def __init__(self, allImageNames, allMaskImageNames, batch_size=100, patch_size=(64, 64)):
        self._batchSize = batch_size
        self._allImageNames = allImageNames
        self._allMaskImageNames = allMaskImageNames
        self._targetSize = patch_size

        self._n = 1
        self._trainLen = len(self._allImageNames)*self._batchSize*self._n

        n = len(self._allImageNames)
        print("Totally", n, "training images")

        self._imageAll = []
        self._maskAll = []

        for it in range(n):
            thisMaskName = self._allMaskImageNames[it]
            thisImageName = self._allImageNames[it]

            #thisImg = io.imread(thisImageName)
            #thisMask = io.imread(thisMaskName)

            thisImg = sitk.ReadImage(thisImageName)
            thisImg = sitk.GetArrayFromImage(thisImg)
            if thisImg.ndim == 3:
                thisImg = np.squeeze(thisImg, axis=0)

            thisMask = sitk.ReadImage(thisMaskName)
            thisMask = sitk.GetArrayFromImage(thisMask)
            if thisMask.ndim == 3:
                thisMask = np.squeeze(thisMask, axis=0)

            thisImg, thisMask = adjustData(thisImg, thisMask)

            self._imageAll.append(thisImg)
            self._maskAll.append(thisMask)
            
    def __getitem__(self, index):

        ImageIndex = index//self._batchSize

        thisImg = self._imageAll[ImageIndex]
        thisMask = self._maskAll[ImageIndex]

        sz = thisImg.shape
        patchShape = self._targetSize

        while 1:

            allPatchTopLeftX = np.random.randint(0, sz[0]-patchShape[0], size=1)
            allPatchTopLeftY = np.random.randint(0, sz[1]-patchShape[1], size=1)

            thisTopLeftX = allPatchTopLeftX[0]
            thisTopLeftY = allPatchTopLeftY[0]

            thisLabelImPatch = np.zeros((2, patchShape[0], patchShape[1]), dtype=np.float32)
            thisImPatch = np.zeros((2, patchShape[0], patchShape[1]), dtype=np.uint16)

            thisLabelImPatch[0, :, :] = thisMask[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]
            thisLabelImPatch[1, :, :] = thisMask[thisTopLeftX:(thisTopLeftX+patchShape[0]), (sz[1]-thisTopLeftY-patchShape[1]):(sz[1]-thisTopLeftY)]

            if thisLabelImPatch[0].max() + thisLabelImPatch[1].max() > 0:
                thisImPatch[0, :, :] = thisImg[thisTopLeftX:(thisTopLeftX+patchShape[0]), thisTopLeftY:(thisTopLeftY+patchShape[1])]
                thisImPatch[1, :, :] = thisImg[thisTopLeftX:(thisTopLeftX+patchShape[0]), (sz[1]-thisTopLeftY-patchShape[1]):(sz[1]-thisTopLeftY)]

                T  = ToTensor()
                
                tensorImagePatch = T(thisImPatch)
                tensorLabelPatch = T(thisLabelImPatch)

                return tensorImagePatch, tensorLabelPatch

    def __len__(self):
        return self._trainLen

    print(targets.shape)
