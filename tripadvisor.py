
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from pandas import DataFrame

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims- list containing the dimensions of each layer in our network
    
    Returns:
    parameters- dictionary containing  parameters W1, b1, ..., WL, bL:
    """
    parameters = {}
    L = len(layer_dims)      
    np.random.seed(3);

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def sigmoid(z):
    """
    Compute the sigmoid of z
    """
    s = 1/(1 + np.exp(-z))
    return s, z

def relu(z):

    return z*(z > 0), z

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A- activations from previous layer
    W- weights matrix
    b- bias vector

    Returns:
    Z- the input of the activation function 
    cache- dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    """
    
    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X- data
    parameters- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches
    
def compute_cost(AL, Y, lambd, L, parameters):
    """
    Implement the cost function 
    """
    
    m = Y.shape[1]

  
    temp_cost = np.dot(Y, (np.log(AL)).T) + np.dot(1-Y, (np.log(1-AL)).T)
    temp_cost = temp_cost * np.eye(np.shape(AL)[0], dtype = 'int64')
    cost = (-1/m) * np.sum(temp_cost)

    reg_cost = 0

    for l in range(1, L):
        reg_cost = reg_cost + np.sum(np.square(parameters['W' + str(l)]))

    cost = cost + (lambd/(2*m)) * reg_cost   
    cost = np.squeeze(cost)    
    return cost


# In[6]:

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
   
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA- post-activation gradient, of any shape
    cache- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    
    return dZ


def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T) + (lambd/m) * W
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db




def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
   
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
     
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid', lambd)
    

    for l in reversed(range(L-1)):
         
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu', lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    """
    L = len(parameters) // 2 # number of layers in the neural network

    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters



def YesNoType(x):
    if x=="YES":
        return 1
    else:
        return 0
    
def str_to_int(str):
    x=0
    for l in str:
        x += ord(l)
    return int(x)

path = ('TripAdvisor.xlsx')
df = pd.read_excel(path)

cols = ['User country', 'Nr. reviews','Nr. hotel reviews','Helpful votes','Score','Period of stay','Traveler type','Swimming Pool','Exercise Room','Basketball Court','Yoga Classes','Club','Free Wifi','Hotel name','Hotel stars','Nr. rooms','User continent','Member years','Review month','Review weekday']

df['Club']=df['Club'].apply(lambda x : YesNoType(x))
df['Exercise Room']=df['Exercise Room'].apply(lambda x : YesNoType(x))
df['Swimming Pool']=df['Swimming Pool'].apply(lambda x : YesNoType(x))
df['Basketball Court']=df['Basketball Court'].apply(lambda x : YesNoType(x))
df['Free Wifi']=df['Free Wifi'].apply(lambda x : YesNoType(x))
df['Yoga Classes']=df['Yoga Classes'].apply(lambda x : YesNoType(x))

cols_2 = ['Period of stay', 'Hotel name', 'User country', 'Traveler type', 'User continent', 'Review month', 'Review weekday']

for y in cols_2:
    df[y]=df[y].apply(lambda x: str_to_int(x)) 
  
temp = df.as_matrix()
"""

"""
print(np.shape(temp))

x = temp[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
y = temp[:, [4]]
x = x.T
test_x = x[:, 0:50]
train_x = x[:, 50:505]
print(np.shape(train_x))
print(np.shape(test_x))

y = y.T
y = y-1;
y = np.array(y, dtype = 'int64')
ohm = np.zeros((504, 5))
ohm[np.arange(504), y] = 1
y = ohm.T    
y = np.array(y, dtype = 'int64')
test_y = y[:, 0:50]
train_y = y[:, 50:505]
print(np.shape(train_y))
print(np.shape(test_y))


layers_dims = [19, 30, 50, 20, 30, 9, 5]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0065, num_iterations = 3000, print_cost=False, lambd = 0.7):#lr was 0.009


    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y, lambd, len(layers_dims), parameters)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambd)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters




parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 7200, print_cost = True, lambd = 0.7)




def predict(X, Y, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
 
    AL, cache = L_model_forward(X, parameters)
    predictions = np.zeros(np.shape(AL))

    for i in range(np.shape(Y)[1]):
        col_max = np.max(AL[:, i])
        predictions[:, i] = (AL[:, i] == col_max)    

    temp_accuracy = np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T)
    temp_accuracy = temp_accuracy * np.eye(np.shape(AL)[0], dtype = 'int64')
    accuracy = (np.sum(temp_accuracy)/(Y.size))*100
    print(accuracy)

    return predictions




print("\n\n")
print("The Accuracy for the training set is : ");
pred_train = predict(train_x, train_y, parameters)
print("\n\n")



print("The Accuracy for the test set is : ")
pred_test = predict(test_x, test_y, parameters)
