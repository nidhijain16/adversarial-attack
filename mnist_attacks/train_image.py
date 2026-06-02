#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().system('pip install adversarial-robustness-toolbox')
get_ipython().system('pip install tensorflow')


# In[39]:


import sys
get_ipython().system('{sys.executable} -m pip install torch')


# In[2]:


import numpy as np
import tensorflow as tf
import keras
#from keras.datasets import cifar10
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import KerasClassifier


# In[6]:


(x_train, y_train), (x_test, y_test) = mnist.load_data() 
#(x_train, y_train, x_test, y_test)
print(x_train.shape)
print(x_test.shape)
x_train.shape[1:]


# In[13]:


#Load CIFAR-10 & MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data() #cifar10.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# Create a neural network model

model = Sequential()
#model.add(Flatten(input_shape=(28,28)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1:])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[14]:


# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[15]:


# reshape data to fit model
#X_train, X_test = x_train/255, x_test/255
# one-hot encode target column
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Re-train the model on the augmented dataset
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
# Evaluate the model's accuracy on the test dataset
score1 = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score1[1])


# In[16]:


# Create a KerasClassifier instance
tf.compat.v1.disable_eager_execution()
classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)


# In[18]:


# Generate adversarial examples
jsma = SaliencyMapMethod(classifier, theta=1, gamma=0.1)
x_train_adv = jsma.generate(x_train)


# In[28]:


x_train_adv = np.concatenate((x_train, x_train_adv))
y_train_adv = np.concatenate((y_train, y_train))


# In[20]:


model.fit(x_train_adv, y_train_adv, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model's accuracy on the test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])


# In[30]:


x_train_adv.shape


# In[32]:


import matplotlib.pyplot as plt
# Pick an image to display
img_num = 4200
img = x_train[img_num]
img1 = x_train_adv[img_num]
# Show the image
plt.imshow(img)
plt.show()


# In[33]:


#img_num = 
#img = x_train[img_num]
#img1 = x_train_adv[img_num]
# Show the image
plt.imshow(img1)
plt.show()


# In[27]:


#4200+5999


# In[ ]:




