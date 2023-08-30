'''
@Author: Madhav Bhandari
'''

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


vgg =  VGG16(input_shape= IMAGE_SIZE + [3], weights= 'imagenet', include_top = False)

for layers in vgg.layers:
    layers.trainable = False
    
folders = glob('/media/maddy/D/Bioinformatics/Datasets/train/*')

x= Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs= vgg.input, outputs= prediction)

model.summary()

model.compile(
              loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/media/maddy/D/Bioinformatics/Datasets/train',
                                                 target_size=(224,224),
                                                 batch_size= 32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('/media/maddy/D/Bioinformatics/Datasets/test',
                                                 target_size=(224,224),
                                                 batch_size= 32,
                                                 class_mode='categorical')

#fit the model

r= model.fit(training_set,
                       validation_data= test_set,
                       epochs=5,
                       steps_per_epoch =len(training_set),
                       validation_steps=len(test_set)
                       )


import tensorflow as tf

from keras.models import load_model

model.save('/media/maddy/D/Bioinformatics/model_vgg16.h5')



plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()
plt.savefig('/media/maddy/D/Bioinformatics/LossVal_loss')


plt.plot(r.history['accuracy'], label='train accuracy')  
plt.plot(r.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
plt.savefig('/media/maddy/D/Bioinformatics/AccVal_acc')




    












