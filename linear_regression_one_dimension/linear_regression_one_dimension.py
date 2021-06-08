import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# We will be using numpy, pandas and matplotlib
# sklearn is imported to show how easy
# it makes things. 

data = pd.read_csv('linear_data.txt') # We read the data
X = data.iloc[:,0]  # Import data into two arrays that we will manioulate
y = data.iloc[:,1] 
m = X.size # This is the number of data we have
data.head() # On jupyter notebook, this shows us the beginning a few lines of the data
X = X[:,np.newaxis] # Now we make the arrays, into column vectors
y = y[:,np.newaxis]


plt.scatter(X, y) # Let us see how our data looks like
plt.xlabel('Feature')
plt.ylabel('Target')
plt.savefig("image_linear_regression_one_dimension_scatter.png")


alpha = 0.01 # We define the coefficient used in gradient descent

def h(theta_0,theta_1,X): # The hypothesis
    return theta_0 + theta_1 * X

def dJ(theta_0,theta_1,X,y): # The derivarives of the cost function 
    dJ0 = np.sum(h(theta_0,theta_1,X)-y)/m
    dJ1 = np.sum(np.multiply(h(theta_0,theta_1,X)-y,X))/m
    return np.array([dJ0,dJ1])

def gradient_descent(theta_0,theta_1,X,y,alpha): # Gradient descent directly implemented as it was explained in the notes on bcivitcioglu.github.io
    iterations = 1500
    for a in range(iterations):
        der_J = dJ(theta_0,theta_1,X,y)
       
        temp0 = theta_0 - alpha * der_J[0]
        temp1 = theta_1 - alpha * der_J[1]
        theta_0 = temp0
        theta_1 = temp1
 
    return np.array([theta_0, theta_1])

theta = gradient_descent(4,4,X,y,alpha) # We run the gradient descend
print(theta) # The theta values of the hypothesis

plt.scatter(X, y) # Now we see how our hypothesis fits
plt.xlabel('Feature')
plt.ylabel('Target')
plt.plot(X, h(theta[0],theta[1],X),color='r')
plt.savefig("image_linear_regression_one_dimension.png")
plt.show()

# End of programme
