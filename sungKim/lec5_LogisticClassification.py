'''
Classification
    Spam Detection : Spam(1) of Ham(0)
    Facebook feed : show(1) or hide(0)
    Credit Card Fraudulent Transaction detection : legitimate(0)/fraud(1)
    
    g(z)=1/(1+e^(-z)) <=== logistic function, sigmoid function
                            0과 1 사이
                            
    정확도가 높다.
    뉴럴 네트워크의 중요한 구성요소
    
    New cost function for logistic
        cost(H(x),y) = -log(H(x))       :y=1
                       -log(1-H(x))     :y=0
                       
        y=1                            y=0
        H(x)=1 -> cost(1)=0            H(x)=0 -> cost=0
        H(x)=0 -> cost=무한히커진다.    H(x)=1 -> cost=무한히 커진다.
        
        cost(H(x),y) = -y*log(H(x))-(1-y)*log(1-H(x))
'''
import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

# placeholders for a tensor that will be always fed
X=tf.placeholder(tf.float32,shape=[None,2])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis using sigmoid : tf.div(1.,1.+tf.exp(tf.matmul(X,W)+b))
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

#cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Accuracy computation
#True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

#launch graph
with tf.Session() as sess:
    #initalize tensorflow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val,_=sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
        if step %200 == 0:
            print(step,cost_val)
            
    #Accurancy report
    h,c,a = sess.run([hypothesis,predicted,accuracy],
                    feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis: ",h,"\nCorrect(Y): ",c, "\nAccuracy: ", a)