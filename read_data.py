import os
import cv2
from PIL import Image
import numpy as np


def read_data(root):

    # Reading the input images and putting them into a numpy array
    data=[]
    labels=[]

    height = 30
    width = 30
    channels = 3
    classes = 43
    n_inputs = height * width * channels

    for i in range(classes) :
        path = os.path.join(root,'train',str(i))+'/'
        print(path)
        Class=os.listdir(path)
        for a in Class:
            try:
                image=cv2.imread(path+a)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                labels.append(i)
            except AttributeError:
                print(" ")
                
    Cells=np.array(data)
    labels=np.array(labels)

    #Randomize the order of the input images
    s=np.arange(Cells.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    Cells=Cells[s]
    labels=labels[s]

    return Cells, labels