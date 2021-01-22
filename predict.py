import os
import cv2
import math
from tqdm import tqdm
import numpy
import numpy as np
import torch
import torch.nn as nn
import skimage.io as io
import SimpleITK as sitk

from UNet import UNet
import data as D
import getTrainingData as G

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def segmentPatch(model, inputImagePatchArray):
    imageArray = inputImagePatchArray

    testBatchSize = 1
    imageArray = np.reshape(imageArray, (1,)+imageArray.shape)

    model.eval()
    with torch.no_grad():
        dataToPredict = np.expand_dims(imageArray, axis=1)
        TensorToPredict = torch.from_numpy(dataToPredict.astype(dtype=np.float32))
        TensorToPredict = TensorToPredict.to(device)
        results = model(TensorToPredict)

    #results = nn.Sigmoid()(results)

    outputSegArray = results[0, 0, :, :]

    return outputSegArray.cpu().numpy()

def segmentPatchBatch(model, inputImagePatchBatchArray):

    sz = inputImagePatchBatchArray.shape

    testBatchSize = sz[0]
    
    model.eval()
    with torch.no_grad():
        dataToPredict = np.expand_dims(inputImagePatchBatchArray, axis=1)
        TensorToPredict = torch.from_numpy(dataToPredict.astype(dtype=np.float32))
        TensorToPredict = TensorToPredict.to(device)
        results = model(TensorToPredict)

    results = nn.Sigmoid()(results)

    outputSegArray = results[:, 0, :, :]

    return outputSegArray.cpu().numpy()

def segmentImageTraverseNonoverlappingPatch(model, imageArray, patchSideLen=16):
    sz = imageArray.shape

    assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen),"Image shape must be >= " + str(patchSideLen) + "-cubed."

    segArray = numpy.zeros(sz, dtype=numpy.uint8)
    # for itx in tqdm(range(0, sz[0]-patchSideLen, patchSideLen), ncols = 100):
    for itx in range(0, sz[0]-patchSideLen, patchSideLen):
        for ity in range(0, sz[1]-patchSideLen, patchSideLen):
                thisPatch = imageArray[itx:(itx+patchSideLen), ity:(ity+patchSideLen)]
                thisPatchSeg = segmentPatch(model, thisPatch)
                segArray[itx:(itx+patchSideLen), ity:(ity+patchSideLen)] = thisPatchSeg.astype(numpy.uint8)

    return segArray

def segmentImageRandomSample(model, imageArray, patchSideLen = 16, numPatchSample = -1, batch_size = 1):
    sz = imageArray.shape

    assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen),"Image shape must be >= " + str(patchSideLen) + "-cubed."

    if numPatchSample == -1:
        # not given, have to compute here
        numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*30)
        # numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*(sz[2]/patchSideLen)*2)

    print("numPatchSample = ", numPatchSample)

    # this saves the segmentation result
    segArray = numpy.zeros(sz, dtype=numpy.float32)

    patchShape = (patchSideLen, patchSideLen)
    imagePatchBatch = numpy.zeros((batch_size, patchShape[0], patchShape[1]), dtype=numpy.float32)
    for itPatch in tqdm(range(0, numPatchSample, batch_size), ncols=100):

        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]

            imagePatchBatch[itBatch, :, :] = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])]

        segBatch = segmentPatchBatch(model, imagePatchBatch) # imagePatchBatch 已经带上batch size，先假设batchsize = 1

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            segArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])] += segBatch[itBatch, :, :]

    return segArray

def segmentImageRandomSampleDivideSampleDensity(model, imageArray, patchSideLen = 16, numPatchSample = -1, batch_size = 1):
    sz = imageArray.shape

    assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen),"Image shape must be >= " + str(patchSideLen) + "-cubed."

    if numPatchSample == -1:
        # not given, have to compute here
        numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*30)

    print("numPatchSample = ", numPatchSample)

    # this saves the segmentation result
    segArray = numpy.zeros(sz, dtype=numpy.float32)

    patchShape = (patchSideLen, patchSideLen)
    imagePatchBatch = numpy.zeros((batch_size, patchShape[0], patchShape[1]), dtype=numpy.float32)

    # to store the sampling prior
    priorArray = numpy.zeros(sz, dtype=numpy.float32)
    # priorArray = numpy.ones(sz, dtype=numpy.float32)
    patchOne = numpy.ones(patchShape, dtype=numpy.float32)

    for itPatch in tqdm(range(0, numPatchSample, batch_size), ncols=100): # numPatchSample为总数目，batch size为一次预测所需要的数目

        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]

            imagePatchBatch[itBatch, :, :] = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])]

        segBatch = segmentPatchBatch(model, imagePatchBatch)
        #segBatch = segmentPatchBatch(model, imagePatchBatch, needThr=False)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            segArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])] += segBatch[itBatch, :, :]

            priorArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])] += patchOne

    priorArray[priorArray==0]=1. #代表为0的位置已做了一次预测，0/1=0
    # segArray /= (priorArray + numpy.spacing(1)) #加上一个极小值
    # segArray  /= priorArray #加上一个极小值
    segArray = (segArray + 1e-6) / priorArray #加上一个极小值
    segArray *= 100
    print(segArray.max())

    return segArray

if __name__ == "__main__":
    model_ckpt_path = './checkpoint/model_spect_210113_A_bce-loss_64/1693.pth'

    print(f'using device:{device}')
    model = UNet(in_channels=1, n_classes=1)
    model.load_state_dict(torch.load(model_ckpt_path))
    model = model.to(device)

    predImageDir = '/data1/huangkaibin/BoneScanDataset/val/predImg/'

    for imageName in os.listdir(predImageDir):
        print(imageName)
        newImageName = './output_64/' + imageName.replace('dcm', 'png')
        imageName = predImageDir+imageName

        img = sitk.ReadImage(imageName)
        img = sitk.GetArrayFromImage(img)
        img = np.squeeze(img, axis=0)

        img = img.astype(np.float)/255.0

        # change patch size here
        predictArr = segmentImageRandomSampleDivideSampleDensity(model, img, patchSideLen = 64)

        cv2.imwrite(newImageName, predictArr)


