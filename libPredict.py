################################################################################
# This only uses 2D RGB image as input and output 1-channel 2D
# image. i.e., NOT multiple channel.
#
# This is adapted from the code that is able to make multiple-channel.
################################################################################


import math
import sys
import os

import numpy
import numpy.random

import cv2
import skimage.io
import skimage.transform

import torch


def segment2DRGBPatch(model, inputImagePatchArray, inputImageIsChannelLast = True):

    # This segments ONE RGB 2D patch. The input inputImagePatchArray
    # is a (patchSideLen, patchSideLen, 3) numpy array. Next, this
    # needs to be reshaped to (#batches, #channels, patchSideLen,
    # patchSideLen) where #batches=1 and #channels=3 in this case

    if inputImageIsChannelLast:
        imageArray = numpy.transpose(inputImagePatchArray.astype(numpy.float32), (2, 0, 1))

    print("imageArray.shape = ", imageArray.shape)
    numChannel = 3
    #assert(imageArray.shape == (patchSideLen, patchSideLen, numChannel)), "Image shape must be >="+str(patchSideLen) + "-cubed bt got "

    imageArray = numpy.reshape(imageArray, (1,) + imageArray.shape) # add a dim to fit torch's requirement: 1st is sample dim

    imageArray = torch.from_numpy(imageArray)

    with torch.no_grad():
        results = model(imageArray)

    results = results.numpy()

    # print(results.dtype)
    # print(results.shape)
    # print(results.max())
    # print(results.min())

    results[results<=0] = 0

    outputSegArray = 200*results[0, 1, :, :]

    return outputSegArray

def segment2DRGBPatchBatch(model, inputImagePatchBatchArray, inputImageIsChannelLast = True):
    # This segments a list of RGB 2D patches. The input
    # inputImagePatchArray is a (#batches, numChannel, patchSideLen,
    # patchSideLen) numpy array. Next this needs to be reshaped to
    # (#batches, #channels, patchSideLen, patchSideLen) where and
    # #channels=3 in this case

    sz = inputImagePatchBatchArray.shape

    if inputImageIsChannelLast:
        # torch needs chanel first, need to transpose if the input array has channel last
        inputImagePatchBatchArray = numpy.transpose(inputImagePatchBatchArray.astype(numpy.float32), (0, 3, 1, 2))

    inputImagePatchBatchArray = torch.from_numpy(inputImagePatchBatchArray)

    with torch.no_grad():
        results = model(inputImagePatchBatchArray)

    outputSegBatchArray = results.numpy()

    if inputImageIsChannelLast:
        # torch needs chanel first, need to transpose if the input array has channel last
        outputSegBatchArray = numpy.transpose(outputSegBatchArray, (0, 2, 3, 1))


    outputSegBatchArray = outputSegBatchArray[:, :, :, 0]
    

    print("*************  ", outputSegBatchArray.shape)
    o = outputSegBatchArray[1, :, :]
    print(o.shape)
    cv2.imwrite('a.png', o*256)

    return outputSegBatchArray


def segment2DRGBImageTraverseNonoverlappingPatch(model, imageArray, patchSideLen = 64, inputImageIsChannelLast = True):
    sz = imageArray.shape

    assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen),"Image shape must be >= " + str(patchSideLen) + "-cubed."

    segArray = numpy.zeros((sz[0], sz[1]), dtype=numpy.float32)
    for itx in range(0, sz[0]-patchSideLen, patchSideLen):
        for ity in range(0, sz[1]-patchSideLen, patchSideLen):
            thisPatch = imageArray[itx:(itx+patchSideLen), ity:(ity+patchSideLen)]
            thisPatchSeg = segment2DRGBPatch(model, thisPatch, inputImageIsChannelLast)
            segArray[itx:(itx+patchSideLen), ity:(ity+patchSideLen)] = thisPatchSeg

    return segArray


# def segment2DRGBImageRandomSample(model, imageArray, patchSideLen = 64, numPatchSampleFactor = 10, batch_size = 1, num_segmetnation_classes = 2, inputImageIsChannelLast = True):
#     sz = imageArray.shape
#     numChannel = 3 # for RGB

#     assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen and sz[2] == 3),"Image shape must be >= " + str(patchSideLen) + "-cubed."
#     if sz[2] != numChannel:
#         print("Only process RGB image")
#         exit(-1)

#     # the number of random patches is s.t. on average, each pixel is
#     # sampled numPatchSampleFactor times. Default is 10
#     numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*numPatchSampleFactor)

#     #print("numPatchSample = ", numPatchSample)

#     # this saves the segmentation result
#     segArray = numpy.zeros((sz[0], sz[1], num_segmetnation_classes), dtype=numpy.float32)

#     patchShape = (patchSideLen, patchSideLen, numChannel)
#     imagePatchBatch = numpy.zeros((batch_size, patchShape[0], patchShape[1], numChannel), dtype=numpy.float32)

#     for itPatch in range(0, numPatchSample, batch_size):

#         allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
#         allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)

#         for itBatch in range(batch_size):
#             thisTopLeftX = allPatchTopLeftX[itBatch]
#             thisTopLeftY = allPatchTopLeftY[itBatch]

#             imagePatchBatch[itBatch, :, :, :] = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1]), :]

#         segBatch = segment2DRGBPatchBatch(model, imagePatchBatch, inputImageIsChannelLast)

#         for itBatch in range(batch_size):
#             thisTopLeftX = allPatchTopLeftX[itBatch]
#             thisTopLeftY = allPatchTopLeftY[itBatch]

#             segArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1]), :] += segBatch[itBatch, :, :, :]

#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # segArray contains multiple channels of output. The 1-st is the
#     # corresponding output for the 1st object.
#     outputSegArrayOfObject1 = segArray[:, :, 1]

#     return outputSegArrayOfObject1

def segment2DRGBImageRandomSampleDividePrior(model, imageArray, patchSideLen = 64, numPatchSampleFactor = 10, batch_size = 1, num_segmetnation_classes = 1, inputImageIsChannelLast = True):
#def segment2DRGBImageRandomSampleDividePrior(model, imageArray, patchSideLen = 64, numPatchSampleFactor = 10, batch_size = 1, num_segmetnation_classes = 3):
    sz = imageArray.shape
    #numChannel = 3 # for RGB
    numChannel = 1 # for Gray

    #assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen and sz[2] == 3),"Image shape must be >= " + str(patchSideLen) + "-cubed."
    assert(sz[0] >= patchSideLen and sz[1] >= patchSideLen),"Image shape must be >= " + str(patchSideLen) + "-cubed."
    '''
    if sz[2] != numChannel:
        print("Only process RGB image")
        exit(-1)
    '''
    # the number of random patches is s.t. on average, each pixel is
    # sampled numPatchSampleFactor times. Default is 10
    numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*numPatchSampleFactor)

    print("numPatchSample = ", numPatchSample)

    # this saves the segmentation result
    segArray = numpy.zeros((sz[0], sz[1], num_segmetnation_classes), dtype=numpy.float32)
    priorImage = numpy.zeros((sz[0], sz[1]), dtype=numpy.float32)

    patchShape = (patchSideLen, patchSideLen, numChannel)
    imagePatchBatch = numpy.zeros((batch_size, patchShape[0], patchShape[1], numChannel), dtype=numpy.float32)

    for itPatch in range(0, numPatchSample, batch_size):

        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]

            #imagePatchBatch[itBatch, :, :, :] = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1]), :]
            imagePatchBatch[itBatch, :, :, 0] = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])]

        segBatch = segment2DRGBPatchBatch(model, imagePatchBatch)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]

            #segArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1]), :] += segBatch[itBatch, :, :, :]
            segArray[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1]), 0] += segBatch[itBatch, :, :]
            priorImage[thisTopLeftX:(thisTopLeftX + patchShape[0]), thisTopLeftY:(thisTopLeftY + patchShape[1])] += numpy.ones((patchShape[0], patchShape[1]))

    for it in range(num_segmetnation_classes):
        segArray[:, :, it] /= (priorImage + numpy.finfo(numpy.float32).eps)
        segArray[:, :, it] *= 100

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # segArray contains multiple channels of output. The 1-st is the
    # corresponding output for the 1st object.
    outputSegArrayOfObject1 = segArray[:, :, 0]

    return outputSegArrayOfObject1


