import io
import numpy
from tensorflow.keras.models import load_model
import cv2

def get_model():

    return load_model('Model.h5')

def get_tensor(image_bytes):
    from PIL import Image
    img= Image.open(io.BytesIO(image_bytes))
    imcv = numpy.asarray(img.convert('L'))
    #open_cv_image = numpy.array(img)
    #img = open_cv_image[:, :, ::-1].copy()
    #grayimage = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
    #imcv = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2GRAY)
    #grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    (thresh,Image) = cv2.threshold(imcv, 127, 255, cv2.THRESH_BINARY)
    Img = cv2.resize(Image,(100,100),interpolation = cv2.INTER_AREA)
    img = Img.reshape(1,100,100,1)
    return img
   # image = Image.open(io.BytesIO(image_bytes))
   # image = cv2.resize(image,(100,100),interpolation= cv2.INTER_AREA)
   # return image





#originalImage = cv2.imread('./N1.jpeg').astype('uint8')
#grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
  
#(thresh,Image) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
#Img = cv2.resize(Image,(100,100),interpolation = cv2.INTER_AREA)
#img = Img.reshape(1,100,100,1)