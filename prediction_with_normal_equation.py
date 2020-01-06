import numpy as np
import matplotlib.pyplot as plt

a = np.array((1,2,3,4))
print(np.c_[np.ones((4, 1)), a])

_tt = 2 * np.random.rand(100, 1)  

def generate_data():  
    X = 2 * np.random.rand(100, 1)
    _X = np.c_[np.ones((100, 1)), X] 
    y = 3 * X + np.random.randn(100, 1) + 4
      
    # y = 4 + 3X + Gaussian noise  
    # theta_0 or Bias Term = 4   
    # theta_1 = 3  
      
    return _X, y  

def generate_noiseless_data():  
    X = np.random.rand(100, 1) # * 2
    y = 3 * X  # + 4
      
    # y = 4 + 3X  
    # theta_0 or Bias Term = 4   
    # theta_1 = 3  
      
    return X, y  

def get_best_param(X, y):  
    X_transpose = X.transpose() 
    # For 2-D arrays, .dot() it is the matrix product
    # res = np.matmul(X_transpose, X) # X_transpose.dot(X)
    res = X_transpose.dot(X) 
    inv = np.linalg.inv(res)
    # best_params = np.linalg.inv(res).dot(X_transpose).dot(y)  
    best_params = inv.dot(X_transpose).dot(y)  
    # normal equation  
    # theta_best = (X.T * X)^(-1) * X.T * y  
      
    return best_params # returns a list  

X, y = generate_data() #  generate_noiseless_data() 
plt.plot(X[:,1],  # draw only not-bias column
         y, "b.") 

params = get_best_param(X, y) 
print(f"W: {params}")

test_X = np.array([[0], [2]])  
test_X_b = np.c_[np.ones((2, 1)), test_X] 
# _X = np.c_[np.ones((2, 1)), X] 

prediction = test_X_b.dot(params)
print(f"Prediction: {prediction}") 
plt.plot(test_X, prediction, "r--") 

plt.show()