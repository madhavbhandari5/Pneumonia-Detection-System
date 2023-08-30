'''
@Author: Madhav Bhandari
'''

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input

model = load_model('/media/maddy/D/Bioinformatics/model_vgg16.h5')

'''
img = image.load_img('/media/maddy/D/Bioinformatics/val/NORMAL2-IM-1438-0001.jpeg', target_size = (224,224))
x =  image.image_to_array(img)
x= np.expand_dims(x, axis=0)

img_data = preprocess_input(x)
classes = model.predict(img_data)


'''