'''
Regression(data)
X(feature data) | Y
1               |1
2               |2
3               |3
(Linear)Hypothesis(선형가설)
    1. H(x) = Wx + b
    2. which hypothesis is better?
        Cost(loss) function : How fit the line to our (training)data
        (H(x)-y)^2 --> give more panalty when it get more gap!
                       you don t need to consider about sign(minus/plus)
                       
        for i in len(data):
            cost = ((H(x(i))-y(i))**2)/len(data)
            
        Goal : Minimize cost
            cost<=cost(W,b)
        
'''
'''
1.Build graph using TensorFlow operations
2.feed data and run graph(operation)
    sess.fun(op,feed_dict={x:x_data})
3.update variables in the graph(and return values)
'''
import tensorflow as tf

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
# 텐서플로우가 변경시키는 값이다.(학습시킬 수 있는 변경값)

# Our hypothesis XW+b
hypothesis = x_train*w + b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean : 평균

#Minimize(GradientDescent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Lanch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(cost),sess.run(w),sess.run(b))


'''
Placeholders
'''

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b 
# + provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Now we can use X and Y in place of x_data and y_data
# placeholders for a tensor that will be always fed using feed_dict
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#Our hypothesis XW+b
hypothesis = X*W+b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Lanch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
#...
for step in range(2001):
    cost_val,W_val,b_val,_= sess.run([cost,W,b,train],
                                     feed_dict={X:[1,2,3,4,5],
                                                Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step,cost_val,W_val,b_val,_)