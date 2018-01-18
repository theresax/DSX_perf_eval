import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Want TensorFlow to not allocate memory for "all of the GPUs"
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

points_n = 2000
clusters_n = 5
iteration_n = 100

#device_name = "/gpu:0"
device_name = "/cpu:0"
with tf.device(device_name): 
    #scenario 1: with generated points
    #points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
    #centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

    #scenario 2: with 500 KB data set    
    pd_1 = pd.read_csv('/user-home/1001/data/Finance-50-16-8_v4.csv')
        
    #scenario 3: with 1 GB data set
    #pd_1 = pd.read_csv('/user-home/1001/data/gdelt1gb.csv', header=None, index_col=0)
    
    #scenario 4: with 15GB data set (CPU only)
    #pd_1 = pd.read_csv('/user-home/1001/data/gdelt-skgm-300-16-8_v2.csv', header=None, index_col=0)

    df_1 = pd_1.as_matrix()
    df_ph = tf.placeholder(tf.float64, shape=pd_1.shape)
    points = tf.get_variable("points", shape=pd_1.shape, dtype=tf.float64, initializer=tf.zeros_initializer())
    centroids = tf.get_variable("centroids", shape=[clusters_n, pd_1.shape[1]], dtype=tf.float64, initializer=tf.zeros_initializer())
    
    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids.initialized_value(), 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)

    assignments = tf.to_int32(assignments)
    partitions = tf.dynamic_partition(points, assignments, clusters_n)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)

    update_centroids = tf.assign(centroids, new_centroids)
    
    init = tf.global_variables_initializer() 
	
# Want TensorFlow to not allocate "all of the memory" for the GPUs visible to it
from keras import backend as K
config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True

def run_kmeans():
    startTime = datetime.now()    
    #NUM_THREADS = 88
    #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:    
    #with tf.Session(config = tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=config) as sess:    

        sess.run(init)
        sess.run(points.assign(df_ph), feed_dict={df_ph: df_1})
        sess.run(centroids.assign(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1])))

        startTime2 = datetime.now()    
        for step in xrange(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
        print("Execution time taken:", datetime.now() - startTime2)   
        #print "centroids" + "\n", centroid_values

    print("Total time taken:", datetime.now() - startTime)  

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    plt.show()	
	
import multiprocessing

# execute code with extra process so that at the end of the process the memory is released
p = multiprocessing.Process(target=run_kmeans)
p.start()
p.join()	
    