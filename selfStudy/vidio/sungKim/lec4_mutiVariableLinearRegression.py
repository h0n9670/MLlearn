'''
Cost function
    H(x1,x2,x3)=w1x1+w2x2+w3x3+b

Matrix multiplication(Dot Product)
    (x1,x2,x3).dot(w1,w2,w3)=w1x1+w2x2+w3x3
    X=(x1,x2,x3)
    W=(w1,w2,w3)
    H(X)=XW
    
Many x instances
    ex) x1=[73,93,89,96,73]
Hypothesis using matrix
    |x11 x12 x13|       |w1|    |x11*w1+x12*w2+x13*w3|
    |x21 x22 x23|   .   |w2| =  |x21*w1+x22*w2+x23*w3|
                        |w3|
'''
import tensorflow as tf
x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

#placeholders for a tensor that will be always fed
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]),name='weight1')
w2 = tf.Variable(tf.random_normal([1]),name='weight2')
w3 = tf.Variable(tf.random_normal([1]),name='weight3')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = w1*x1+w2*x2+w3*x3+b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#launch the graph in as session
sess = tf.Session()
#Initializes global_variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost, hypothesis, train],
                                 feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n",hy_val)

'''Matrix'''

x_data = [[73.,90.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

#placeholders for a tensor that will be alwys fed
X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis
hypothesis = tf.matmul(X,W)+b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#launch the graph in as session
sess = tf.Session()
#Initializes global_variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost, hypothesis, train],
                                 feed_dict={X:x_data,Y:y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n",hy_val)
        
'''loading data from file'''

import numpy as np
xy = np.loadtxt('data_01-test-score.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

#Make sure the shape and data are OK
print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

#placeholders for a tensor that will be alwys fed
X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis
hypothesis = tf.matmul(X,W)+b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#launch the graph in as session
sess = tf.Session()
#Initializes global_variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(12001):
    cost_val,hy_val,_ = sess.run([cost, hypothesis, train],
                                 feed_dict={X:x_data,Y:y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n",hy_val)
        
#Ast my score
print("Your score will be ", sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))
print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X:[[60,70,110],[90,100,80]]}))

'''
Queue Runners
1.파일들의 리스트 만들기
2.읽어 올 리터를 선택,key value분해
3.
'''
import tensorflow as tf
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'],
                                                shuffle=False,name='filename_queue')
reader = tf.TextLineReader()
key,value = reader.read(filename_queue)

#Default values, in case of empty columns. Also specifies the type of the
#decoded result
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

#collect batches of csv in
train_x_batch, train_y_batch=tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10)

#placeholders for a tensor that will be alwys fed
X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis
hypothesis = tf.matmul(X,W)+b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for step in range(2001):
    x_batch,y_batch = sess.run([train_x_batch,train_y_batch])
    cost_val,hy_val,_ = sess.run([cost, hypothesis, train],
                                 feed_dict={X:x_batch,Y:y_batch})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n",hy_val) 
         
coord.request_stop()
coord.join(threads)