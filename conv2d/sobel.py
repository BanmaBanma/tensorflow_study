import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf





myimg = mpimg.imread('lena.jpg')
#528,532,3
#shape = myimg.shape
plt.imshow(myimg)
plt.axis('off')
plt.show()
img = np.reshape(myimg,[1,528,532,3])
img.astype('float32')
imginput = tf.placeholder(tf.float32,[1,528,532,3])
filt = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))

op = tf.nn.conv2d(imginput,filt,strides=[1,1,1,1],padding='SAME')
out = tf.cast( ((op-tf.reduce_min(op))/tf.reduce_max(op)-tf.reduce_min(op))*255, tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    o = sess.run(out,feed_dict={imginput:img})
    o = np.reshape(o,[528,532])
    
    plt.imshow(o)
    plt.axis('off')
    plt.show()
    mpimg.imsave("lena_sobel.jpg",o)