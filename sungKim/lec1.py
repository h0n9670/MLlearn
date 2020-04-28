'''
What is ML
    : Automatic driving을 구현하기 위해서는 조건들이 너무 많이 필요하다.
    => 개발자가 정하지 않고 기계가 스스로 조건들을 학습하는 방법을 생각함

Supervised learning
    : learning with labeled examples - training set
    Most common problem type in ML
    ex)distinguish cat and dog
       Predicting final exam score
       Email filter
    Training data set
       
Unsupervised learning:un-labeled data
    - Google news grouping
    - Word clustering
    
Types of supervised learning
    1.Predicting final exam score based on time spent
        - regression
        
    2. Pass/non-pass based on time spent
        - binary classification(choose one type between pass and non-pass)
    3. Letter grade (A,B,C,E and F) based on time spent
        - multi-label classification
            
'''
import tensorflow as tf

graph = tf.Graph() #그래프 객체 생성

with graph.as_default(): #그래프 사용 선언
    my_x = tf.constant(3) #상수형 노드선언
    my_y = my_x ** 2 # 노드의 연산
    
x = tf.constant(3)
y = x**2

sess = tf.Session()#세션 생성
print(sess.run(x))# 세션 실행
print(sess.run(y))
sess.close()#세션 종료

with tf.Session() as sess: #with사용시 종료가 없어도 된다.
    out = sess.run([x, y])
    print(out)