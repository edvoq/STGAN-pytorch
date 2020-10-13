import numpy as np
import math
from PIL import Image
import torch
import torch.nn.functional as F

TESTSIZE = 344
inputImg = Image.open('YOURIMAGEHERE')

newWidth = TESTSIZE
newHeight = TESTSIZE
if inputImg.height < inputImg.width:
   newHeight = TESTSIZE
   newWidth = TESTSIZE * inputImg.width / inputImg.height
else:
   newWidth = TESTSIZE
   newHeight = TESTSIZE * inputImg.height / inputImg.width

inputImg = inputImg.resize((int(newWidth),int(newHeight)),Image.BICUBIC)
deltaX = newWidth - TESTSIZE
deltaY = newHeight - TESTSIZE
if deltaX > 0 or deltaY > 0:
  leftX = int(deltaX / 2)
  rightX = deltaX - leftX
  topY = int(deltaY /2)
  bottomY = deltaY - topY
  inputImg = inputImg.crop((leftX,topY,newWidth-rightX,newHeight-bottomY))

inputArray = np.asarray(inputImg)
inputArray = (inputArray / 127.5) -1
inputArray = inputArray.transpose((2,0,1))
inputTensor = torch.as_tensor(inputArray,dtype=torch.float)
inputTensor = inputTensor.unsqueeze(0)

model = torch.jit.load("YOUTMODELHERE.pt")
output = model(inputTensor)
# pred = F.softmax(output,dim=1)
# numpy_data = pred[0].detach().cpu().numpy()
# maxval = np.amax(numpy_data,axis=0)
# idx = np.zeros((numpy_data.shape[1],numpy_data.shape[2]),dtype=np.uint8)
# for n in range(numpy_data.shape[0]):
#    for y in range(numpy_data.shape[1]):
#       for x in range(numpy_data.shape[2]):
#          if (maxval[y][x] == numpy_data[n][y][x]):
#               idx[y][x] = n

