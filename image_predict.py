'''
@Author: Madhav Bhandari
'''

from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
from keras.applications.vgg16 import preprocess_input

model = load_model('/media/maddy/D/Bioinformatics/model_vgg16.h5')

#img = load_img('/media/maddy/D/Bioinformatics/val/NORMAL/NORMAL2-IM-1437-0001.jpeg', target_size = (224,224))

img = load_img('/media/maddy/D/Bioinformatics/val/PNEUMONIA/person1949_bacteria_4880.jpeg', target_size = (224,224))
x =  img_to_array(img)
x= np.expand_dims(x, axis=0)

img_data = preprocess_input(x)
classes = model.predict(img_data)

if classes.all() == 0:
    print("The report is Normal.")
elif classes.all() == 1:
    print("The report shows that the person is infected with pneumonia.")
else:
    print("It is not clear yet.")

