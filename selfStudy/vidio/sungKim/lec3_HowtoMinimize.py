'''
What cost(W) looks like?
    Gradient descent algorithm(경사를 따라 감소하는 알고리즘)
        it is used many minimizaion problems
        For a given cost function cost(W,b),it will find W,b to minimize cost
        it can be applied to more general function
        
        How it works?
        How would you find the lowest point?
            1. Start with initial guesses
                - Start at 0,0(or any other value)
                - Keeping changing W and b alittle bit to try and reduce cost(W,b)
            2. Each time you change the parameters, you select the gradient which
               reduces cost(W,b) the most possible
            3. Repeat
            4. Do so until you converge to a local minimum
            5. Has an interesting property
                - Where you start can determine which minimum you end up
                  
            미분을 통한 기울기를 가지고 가장 작은 기울기를 찾아가는 방법
            if you want to Derivative easily, then here is a web site;Dericative Calculator
            
        Convex function인지 항상 확인하고 사용해야 한다.(0이되는 점이 하나)    
'''