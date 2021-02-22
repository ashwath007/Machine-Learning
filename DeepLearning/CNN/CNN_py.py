#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt


# # Importing the Library

# Importing keras library 

# In[52]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[53]:


tf.__version__


# # Preprocessing the Training Dataset

# - Preprocessing the Training Dataset

# In[54]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[55]:


training_set = train_datagen.flow_from_directory('./dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# # Preprocessing the Testing Dataset

#  - Preprocessing the Testing Dataset

# In[56]:


train_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('./dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# # CNN

# Initialising the CNN

# In[57]:


cnn = tf.keras.models.Sequential()


# STEP 1: Convolution

# In[58]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# STEP 2: Polling

# In[59]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[60]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# STEP 3: Flattening

# In[61]:


cnn.add(tf.keras.layers.Flatten())


# STEP 4: Full Connection

# In[62]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# Activation Function

# In[63]:


cnn.add(tf.keras.layers.Dense(units=128, activation='sigmoid'))


# # Training the CNN

# In[64]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[66]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 20)


# In[ ]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

