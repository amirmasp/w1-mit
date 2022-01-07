# Some basic user defined numpy functions CREATED TO USE IN OTHER MODULS
# To use this module in your code, just import this module in your module as:
# import ex01.py as ex1
import numpy as np

#*******************************************************************************
# Row Vector
# A procedure that takes a list of numbers and returns a 2D numpy array
# representing a row vector containing those numbers.
def rv(value_list):
    return np.array([value_list])

#*******************************************************************************


#*******************************************************************************
# Column Vector
# A procedure that takes a list of numbers and returns a 2D numpy array 
# representing a column vector containing those numbers.
def cv(value_list):
    return np.array([value_list]).T

#*******************************************************************************


#*******************************************************************************
# length
# A procedure that takes a column vector and returns
# the vector's Euclidean length (or equivalently, its magnitude) as a scalar.
def length(col_v):
    '''
    we can think of coding the Euclidean norm as a three step process:
    1.Square each item of the original vector (this can be done with 
        element-wise multiplication, col_v*col_v, or the built-in square
         function, np.square(col_v))
    2.Sum up all the elements in the new vector (this can be done with np.sum())
    3.Take the square root of the previous sum.
    '''
    return np.sqrt((np.sum(np.square(col_v))))

#*******************************************************************************


#*******************************************************************************
# normalize 
# A procedure that takes a column vector and returns
# a unit vector in the same direction.
def normalize(col_v):
    return col_v/length(col_v)
    
#*******************************************************************************


#*******************************************************************************
# indexing
# A procedure that takes a 2D array and returns 
# the final column as a two dimensional array.
def index_final_col(A):
    return A[:,-1:]

#*******************************************************************************


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
    return ((Θ.T @ x) + Θ0)/length(Θ)

#*******************************************************************************


#*******************************************************************************
# 1.3) Code for side of hyperplane
# Write a Python function that takes as input:
# a column vector x, a column vector th that is of the same dimension as x,
# a scalar th0. 
# and returns:
# +1 if x is on the positive side of the hyperplane encoded by (th, th0)
#  0 if on the hyperplane
# -1 otherwise
# The answer should be a 2D array (a 1 by 1). 
def positive(x, Θ, Θ0):
    '''
    Recall the formula for how we determine which side of the hyperplane defined
    by θ and  0θ, a point x lies on:
                        sign(Θ^T @ x + Θ0)// formulla
                        return np.sign((Θ.T @ x) + Θ0) //python code
    Now, given a hyperplane and a set of data points, we can think
    which points are on which side of the hyperplane. This is something we
    do in many machine-learning algorithms, as we will explore soon. 

    '''
    
    return np.sign(signed_dist(x, Θ, Θ0))
#*******************************************************************************


#*******************************************************************************
def main():
    value_list = [22,33,44,55,66,77,88,99]
    M = np.array([value_list, value_list])
    row_vector = rv(value_list)
    print('row vector\n',row_vector, ', shape:', row_vector.shape)

    col_vector = cv(value_list)
    print('column vector\n',col_vector, ', shape:', col_vector.shape)

    length_of_vec = length(col_vector)
    print('length of column vector is a scalar:', length_of_vec)

    normalize_vec = normalize(col_vector)
    print('unit vector normal to the hyperplane specified by a column vector,Θ,\n',
        normalize_vec, ', shape:', normalize_vec.shape)

    print('matrix M\n', M, ', shape:', M.shape)
    print('The final column of matrix M as a two dimensional array:')
    print(index_final_col(M), ', shape:', index_final_col(M).shape)

#*******************************************************************************


#*******************************************************************************
if __name__ == '__main__':
    main() 
#*******************************************************************************




 