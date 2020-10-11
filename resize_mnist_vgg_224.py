
from PIL import Image
import os, sys

def resize(path, target_path):
    dirs = os.listdir(path)
    for curDir in dirs:
        if os.path.isdir(os.path.join(path, curDir)):
            for curFile in os.listdir(os.path.join(path, curDir)):
                if os.path.isfile(os.path.join(path, curDir, curFile)):        
                    im = Image.open(os.path.join(path, curDir, curFile))
                    imResize = im.resize((224,224), Image.ANTIALIAS)
                    if not os.path.isdir(os.path.join(target_path, curDir)):
                        os.mkdir(os.path.join(target_path, curDir))
                    imResize.save(os.path.join(target_path, curDir, curFile), 'JPEG', quality=90)       

resize(
    "/home/arda/Projects/detector/prep/vgg/datasets/mnist/trainingSet", 
    "/home/arda/Projects/detector/prep/vgg/datasets/mnist_224/trainingData"
    )

