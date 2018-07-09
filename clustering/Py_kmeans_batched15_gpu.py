
# coding: utf-8

# In[ ]:


get_ipython().system(u'nvidia-smi')


# In[ ]:


get_ipython().system(u'pwd')


# In[ ]:


get_ipython().system(u'ls -lL /user-home/1001/DSX_Projects/k-means1/')


# In[1]:


import numpy as np
import pandas as pd
import math
import tensorflow as tf

# Want TensorFlow to not allocate memory for "all of the GPUs"
import os

def batched_reduce(total_sz, points, centroids, batch_n):
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    all_distances = None
    batch_sz = int(math.ceil(float(total_sz) / batch_n))
    #print("Total size: %d, batch size: %d" % (total_sz, batch_sz))

    # The resulting shape is [1, pd_1.shape[0], pd_1.shape[1]]
    points_expanded = tf.expand_dims(points, 0)

    # The resulting shape is [clusters_n, 1, pd_1.shape[1]]
    centroids_expanded = tf.expand_dims(centroids.initialized_value(), 1)

    for x in range(batch_n):
        slice_start = x * batch_sz
        slice_end = slice_start + batch_sz
        #print("Batch: %d, slice start: %d, slice end: %d" % (x, slice_start, slice_end))
        if (slice_end < total_sz):
            # Working with points_expanded with shape [1, pd_1.shape[0], pd_1.shape[1]]
            points_exp = tf.slice(points_expanded, [0, slice_start, 0], [1, batch_sz, centroids_expanded.shape[2]])        
        else:
            points_exp = tf.slice(points_expanded, [0, slice_start, 0], [1, total_sz-slice_start, centroids_expanded.shape[2]])

        device_name2= "/gpu:0"
        #device_name2= "/cpu:0"
        with tf.device(device_name2):                
            dist = tf.reduce_sum(tf.square(tf.subtract(points_exp, centroids_expanded)), 2)          

        if (all_distances == None): 
            all_distances = dist
        else:    
            all_distances = tf.concat([all_distances, dist], 1)
        #print(all_distances.shape)

        if (slice_end >= total_sz):
            break
    
    return all_distances


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import math
from datetime import datetime

# Want TensorFlow to not allocate memory for "all of the GPUs"
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# You can increase points_n and interation_n to minic a bigger data set

points_n = 2000
clusters_n = 5
iteration_n = 100
batch_n = 15

#device_name = "/gpu:0"
device_name = "/cpu:0"
with tf.device(device_name):     
    #scenario 1: with generated points
    #points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
    #centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

    #scenario 2: with 500 KB data set    
    #pd_1 = pd.read_csv('/user-home/1002/DSX_Projects/test2/Finance-50-16-8_v4.csv')
    #points = tf.constant(pd_1.as_matrix())    
       
    #scenario 4: with 1 GB data set
    #pd_1 = pd.read_csv('/user-home/1001/DSX_Projects/k-means1/gdelt1gb.csv', header=None, index_col=0)
    
    #scenario 5: with 15GB data set (CPU only)    
    pd_1 = pd.read_csv('/user-home/1001/DSX_Projects/k-means1/gdelt-skgm-300-16-8_v2.csv', header=None, index_col=0)
          
    df_1 = pd_1.as_matrix()
    df_ph = tf.placeholder(tf.float64, shape=pd_1.shape)
    points = tf.get_variable("points", shape=pd_1.shape, dtype=tf.float64, initializer=tf.zeros_initializer())
    centroids = tf.get_variable("centroids", shape=[clusters_n, pd_1.shape[1]], dtype=tf.float64, initializer=tf.zeros_initializer())
               
    # The resulting shape of tf.subtract is [clusters_n, pd_1.shape[0], pd_1.shape[1]]
    # The resulting shape of tf.square keeps the input shape
    # From tf.reduce_sum with reduction_indices=2, the result shape becomes [clusters_n, pd_1.shape[0]]        
    points_expanded = tf.expand_dims(points, 0)        
    centroids_expanded = tf.expand_dims(centroids.initialized_value(), 1)
    #distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    distances = batched_reduce(pd_1.shape[0], points, centroids, batch_n)
    #print(distances.shape)

    assignments = tf.argmin(distances, 0)
    assignments = tf.to_int32(assignments)
    partitions = tf.dynamic_partition(points, assignments, clusters_n)
    
    #device3_name = "/cpu:0"
    device3_name = "/gpu:0"
    with tf.device(device3_name):          
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

    update_centroids = tf.assign(centroids, new_centroids)

    init = tf.global_variables_initializer() 


# In[3]:


# Want TensorFlow to not allocate "all of the memory" for the GPUs visible to it
from keras import backend as K
import matplotlib.pyplot as plt
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
#config.RunOptions.report_tensor_allocations_upon_oom

#from tensorflow.core.protobuf import config_pb2
#with session.Session() as sess:
#    sess.run(c, options=config_pb2.RunOptions(
#        report_tensor_allocations_upon_oom=True))

def run_kmeans():
    startTime = datetime.now()    

    with tf.Session(config=config) as sess:  

        sess.run(init)
        sess.run(points.assign(df_ph), feed_dict={df_ph: df_1})
        sess.run(centroids.assign(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1])))

        startTime2 = datetime.now()   

        # for step in xrange(iteration_n):
        for step in range(iteration_n):    
            [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
        print("Execution time taken:", datetime.now() - startTime2)   
        #print "centroids" + "\n", centroid_values

    print("Total time taken:", datetime.now() - startTime)  

    #plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    #plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    #plt.show()


# In[ ]:


import multiprocessing

# execute code with extra process so that at the end of the process the memory is released
p = multiprocessing.Process(target=run_kmeans)
p.start()
p.join()

