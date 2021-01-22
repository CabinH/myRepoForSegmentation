import os

def getTrainingImageAndMaskNames():

    allTrainingImagePath = "/data1/huangkaibin/BoneScanDataset/train/image/"
    allTrainingMaskImagePath = "/data1/huangkaibin/BoneScanDataset/train/label/"
    
    allTrainingMaskImageNames = [x for x in os.listdir(allTrainingMaskImagePath)]
    allTrainingImageNames = [x.replace("_msk.nrrd", ".dcm") for x in allTrainingMaskImageNames]

    allTrainingMaskImageNames = [os.path.join(allTrainingMaskImagePath, x) for x in allTrainingMaskImageNames]
    allTrainingImageNames = [os.path.join(allTrainingImagePath, x) for x in allTrainingImageNames]

    return allTrainingImageNames, allTrainingMaskImageNames

def getValImageAndMaskNames():

    allValImagePath = "/data/huangkaibin/repoHuangKaibin/src/BoneScanDataset/val/predImg/"
    allValMaskImagePath = "/data/huangkaibin/repoHuangKaibin/src/BoneScanDataset/val/predMsk/"
    
    allValMaskImageNames = [x for x in os.listdir(allValMaskImagePath)]
    allValImageNames = [x.replace("_msk.nrrd", ".dcm") for x in allValMaskImageNames]

    allValMaskImageNames = [os.path.join(allValMaskImagePath, x) for x in allValMaskImageNames]
    allValImageNames = [os.path.join(allValImagePath, x) for x in allValImageNames]

    return allValImageNames, allValMaskImageNames

def getPredImageNames():
    allPredImagePath = "/data1/huangkaibin/BoneScanDataset/pred_img/"

    allPredImageNames = [x for x in os.listdir(allPredImagePath)]

    allPredImageNames = [os.path.join(allPredImagePath, x) for x in allPredImageNames]

    return allPredImageNames


