import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import data as D
import getTrainingData as G
from Models import U_Net, AttU_Net
from losses import BinaryDiceLoss

class Trainer:
    def __init__(self, maxEpoch, trainLoader, model, optimizer, device, savecheckpoint, lossCriterion=nn.BCELoss()):
        self.maxEpoch = maxEpoch
        self.trainLoader = trainLoader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.savecheckpoint = savecheckpoint
        #Weighted Binary CrossEntropy Loss better than Dice Loss better than Binary CrossEntropy Loss in BoneScan Dataset
        self.BCECriterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([500]).to(self.device))
        self.DiceCriterion = BinaryDiceLoss()

    def train(self):
        loss_min = float('inf')
        for epoch in range(0, self.maxEpoch):
            self.model.train()

            saveModelPath = self.savecheckpoint

            if not os.path.exists(saveModelPath):
                os.mkdir(saveModelPath)

            for num, (inputs, targets) in enumerate(self.trainLoader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                #outputs = nn.Sigmoid()(outputs)
                #loss = self.DiceCriterion(nn.Sigmoid()(outputs), targets)
                loss = self.BCECriterion(outputs, targets)

                loss.backward()
                self.optimizer.step()



            print(f'epoch:{epoch}, loss:{loss.item()}')
            if loss.item() < loss_min:

                print(f'loss:{loss.item()}, saving model state: {saveModelPath} ...')
                torch.save(self.model.state_dict(), os.path.join(saveModelPath, str(epoch)+'.pth'))
                loss_min = loss.item()

def main(Trainer):
    
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    patchSideLen = 64
    batch_size = 512

    patchSize = (patchSideLen, patchSideLen)

    #trainDataPath = f"/data1/huangkaibin/BoneScanDataset/train/"

    #print('trainDataPath', trainDataPath)
    
    allImageNames, allMaskNames = G.getTrainingImageAndMaskNames()

    trainDataset = D.createDataset(allImageNames, allMaskNames, batch_size=batch_size, patch_size=patchSize)
    
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device:{device}')
    
    #model = UNet(in_channels=1, n_classes=1)
    model = AttU_Net(img_ch=1, output_ch=1)
    #load pretrain model
    #model_ckpt_path = '/data1/huangkaibin/torchUNet2d/checkpoint/model_spect_128_201230_final/293.pth'
    #model.load_state_dict(torch.load(model_ckpt_path))
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    '''Save Model in ckpt'''
    ckpt = './checkpoint/model_attunet_p128_20210122_wbce/'
    print('checkpoint path', ckpt)

    trainer = Trainer(maxEpoch=500000, trainLoader=trainLoader, model=model, optimizer=optimizer, device=device, savecheckpoint=ckpt)
    trainer.train()


if __name__ == '__main__':
    main(Trainer)
