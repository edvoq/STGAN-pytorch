import cv2
import numpy as np

def crop(inputImg, size):
    newWidth = size
    newHeight = size
    if inputImg.shape[0] < inputImg.shape[1]:
       newHeight = size
       newWidth = size * inputImg.shape[1] / inputImg.shape[0]
    else:
       newWidth = size
       newHeight = size * inputImg.shape[0] / inputImg.shape[1]

    resized = cv2.resize(inputImg, (int(newWidth), int(newHeight)))
    
    diff = int((int(newHeight) - int(newWidth)) / 2)
    return resized[diff:-diff,0:]

video_path = '/home/ed/Documents/work/qualitance/STGAN-pytorch/data/video_data/20201012_162733.mp4'
vc = cv2.VideoCapture(video_path)

can_read, frame = vc.read()
frame_count = 0

while can_read:
    frame = np.rot90(frame)
    frame = crop(frame, 128)
    cv2.imwrite('../data/frame_data/{}/{}.jpg'.format(video_path.split('.')[0].split('/')[-1], frame_count), frame)
    can_read, frame = vc.read()
    frame_count += 1