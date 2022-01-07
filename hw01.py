# MIT course > Week1:Basics > Homework1
import numpy as np
import ex01 as ex1

#*******************************************************************************
# 1.1) General hyperplane, distance to point
# Let p be an arbitrary point in R^d. 
# Give a formula for the signed perpendicular (normal) distance 
# from the hyperplane specified by Θ, Θ0 to this point p.
# Answer: (Θ^T p + Θ0) / ||Θ||
# 1.2) Code for signed distance!
# Write a Python function using numpy operations (no loops!) that
# takes column vectors (d x 1) X and Θ (of the same dimension) 
# and scalar Θ0 and returns the signed perpendicular distance
# (as a 1 by 1 array) from the hyperplane encoded by (Θ, Θ0) to X
def signed_dist(x, Θ, Θ0):
    # we use the formulla we found in 1.1)
    # return 1 by 1 matrix of signed distance
    return ((Θ.T @ x) + Θ0)/ex1.length(Θ)

#*******************************************************************************


#******************************************************************************* 
def process_data(data, Θ, Θ0):
    '''
    1.4) operating on data.
    This function takes data and th, th0 and extracts the datapoints in data
    by slicing through each column as a d x 1.  

    '''
    # data is a 2 x 5, means we have 5 datapoints in 2 dimensions
    # we can get its d x n using a tuple (d, n)
    (d, n) = data.shape
    print('\ndata shape is:' ,d,'x',n)
    datapoint_1 = data[:,:1] # slicing first column from data
    print('\ndatapoint 1 is\n', datapoint_1, datapoint_1.shape)
    datapoint_2 = data[:,1:2] # slicing second column from data
    datapoint_3 = data[:,2:3]
    datapoint_4 = data[:,3:4]
    datapoint_5 = data[:,-1:]
    print('\ndatapoint 2 is\n', datapoint_2, datapoint_2.shape)
    print('\ndatapoint 3 is\n', datapoint_3, datapoint_3.shape)
    print('\ndatapoint 4 is\n', datapoint_4, datapoint_4.shape)
    print('\ndatapoint 5 is\n', datapoint_5, datapoint_5.shape)
    # Each datapoin_x is a 2 x 1 column vector,
    p1 = ex1.positive(datapoint_1, Θ, Θ0)
    p2 = np.sign((Θ.T @ datapoint_2) + Θ0)
    p3 = np.sign((Θ.T @ datapoint_3) + Θ0)
    p4 = np.sign((Θ.T @ datapoint_4) + Θ0)
    p5 = np.sign((Θ.T @ datapoint_5) + Θ0)
    print('\np1\n',p1, p1.shape)
    value_list =  [p1[0,0], p2[0,0], p3[0,0], p4[0,0], p5[0,0]]
    print('\nvalue_list\n',value_list, len(value_list))

    # return 2-D array has to be a numpy row vector
    return ex1.rv(value_list)

#*******************************************************************************


#*******************************************************************************
# 1.5) Score
def score(data, labels, th, th0):
    '''
    Write a procedure that takes as input
        data: a d by n array of floats (representing n data points in d dimensions)
        labels: a 1 by n array of elements in (+1, -1), representing target labels
        th: a d by 1 array of floats that together wit
        th0: a single scalar or 1 by 1 array, represents a hyperplane
    and returns the number of points for which the label is equal to the output of
    the positive function on the point.
    Since numpy treats False as 0 and True as 1,
    you can take the sum of a collection of Boolean values directly.

    '''
    # 1.4) is embedded into score function
    # 1. A should be a 1 by 5 array of values, either +1, 0 or -1, indicating,
    # for each point in data, whether it is on the positive side of the hyperplane
    # defined by th, th0.
    A = process_data(data, th, th0) # we are processing the data to see whether 
                                     # datapoints are in which side of hyperplane
    print('\nA\n', A, A.shape )
    print ('labels\n', labels, labels.shape)
    # 2. B hould be a 1 by 5 array of boolean values, either
    # True or False, indicating for each point in data and corresponding 
    # label in labels whether it is correctly classified by 
    # hyperplane th and th0
    B = A == labels 
    print('\nB\n',B, B.shape, ', score:',np.sum(B))
    # we want to count the number of “True”s (correctly classified points) in B
    return np.sum([B])

#*******************************************************************************


#*******************************************************************************
def main():
    # We define data to be a 2 by 5 array (two rows, five columns) of scalars.
    data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
    # It represents 5 data points in two dimensions.
    # We also define labels to 
    labels = ex1.rv([-1, -1, +1, +1, +1])
    # be a 1 by 5 array (1 row, five columns) of 1 and -1 values.
    
    labels = ex1.rv([-1, -1, +1, +1, +1])
    print('data\n',data, data.shape)
    print('\nlabels\n',labels, labels.shape)
    th = ex1.cv([1,1])
    th0 = -2
    print("\nwe defined our hyperplane by random numbers of Θ, Θ0:")
    print('Θ =\n', th, th.shape)

    sco = score(data, labels,th,th0)
    print('score is: ', sco, sco.shape)






    
    
#*******************************************************************************
    

#*******************************************************************************
if __name__ == '__main__':
    main()  
#*******************************************************************************    