import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.metrics import zero_one_loss
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.naive_bayes import GaussianNB


mnist_ip = input_data.read_data_sets('MNIST_data', one_hot=False)
training_images = np.array(mnist_ip.train.images)
'''
print(training_images.shape)
'''
training_labels = np.array(mnist_ip.train.labels).reshape([-1,1])
test_images = np.array(mnist_ip.test.images)
test_labels = np.array(mnist_ip.test.labels).reshape([-1,1])

for i in range(training_images.shape[0]):
    plot.imshow(training_images[i].reshape([28,28]), cmap = plot.get_cmap('gray'));

training_images2 = np.concatenate((training_images, training_labels), axis=1)

df_training = pd.DataFrame(training_images2)

training_mean = df_training.groupby(784).mean()

for i in range(training_mean.shape[0]):
    fig = plot.imshow(np.array(training_mean[i:i]).reshape([28,28]));    
    plot.show()
    plot.draw()
    

training_sd = df_training.groupby(784).std()

for i in range(training_mean.shape[0]):
    fig = plot.imshow(np.array(training_sd[i:i]).reshape([28,28]));   
    plot.show()
    plot.draw()
    
#Question 2   
#naive-bayes
model = GaussianNB()
model.fit(training_images, training_labels.ravel())
predicted= model.predict(test_images)

'''
print(predicted.shape)
error = zero_one_loss(test_labels,predicted)
print("error: %f" %(error) )
accuracy = 1 - error;
print("Probability of accuracy:%f" %(accuracy))
'''
counter = 0;
for i in range(0,10000):
   if predicted[i] not in test_labels[i]:
       counter = counter + 1
      
P_error = (counter/10000)        
print("Probablity of Error: %f"%(P_error))
P_accuracy = 1 - P_error
print("Accuracy rate : %f" %(P_accuracy))

       