# n x m  numpy array is a 2-D array.
#  np.array([[2, 6, 9]]) is a 2-D array with shape: (1, 3) 
# Using one set of square brackets creates a 1-dimensional array,
#  np.array([1,2,3]) is a 1-d array with shape:(3,)
import numpy as np
from numpy.core.fromnumeric import shape


def main():
    # Generating some row and column vector in numpy library
    # Generating a 3×1 numpy array (vector) using three methods
    v = np.array([[1],[5],[3]])
    print('vector v shape is:', v.shape)
    # print(np.dot(v,v)) # this is not work 
    print(v)
    v = np.transpose([[1,5,3]])
    print('same column vector is created by transposing a row vector [1,5,3]\n', v) 
    v = np.array([[1,5,3]]).T
    print('It is often more convenient to use the array attribute .T \n', v)
    print("**********************************************************************")
    
    # Generating a 1×8 numpy array
    one_dimensional_array = np.array([[1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5]])
    print('\nGenerating a 1x8 numpy row array:', one_dimensional_array.shape, ', len =',len(one_dimensional_array))
    print(one_dimensional_array)
    print("**********************************************************************")

    # Generating a 8x1 numpy array
    eight_dimensional_array = np.array([[1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5]]).T
    print('Generating a 8x1 numpy column array:', eight_dimensional_array.shape, ',len =',len(eight_dimensional_array))
    print(eight_dimensional_array)
    print("**********************************************************************")


    # Generating a 3x2 numpy array
    two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
    print('Generating a 3x2 numpy array:', two_dimensional_array.shape, ',len =',len(eight_dimensional_array))
    print(two_dimensional_array)
    print("**********************************************************************")

    # Generating a matrix using sequence of integers [a,b), np.arange(),
    sequence_of_integers = np.arange(5, 12)
    v = np.array([sequence_of_integers])
    print('\n\nnp.arange() returns a list that is not an vector yet\n',sequence_of_integers)
    print(sequence_of_integers.shape)
    print('we can use a sequenced list to create a 1 x d row vector:\n',v)
    print(v.shape)
    print('we can use a sequenced list to create a d x 1 column vector:\n',v.T)
    print(v.T.shape)
    print("**********************************************************************")

    # Generating and populating matrix using Random int Numbers
    # numpy methods return a list, <class 'numpy.ndarray'>, but if we need to create matrices
    rand_int_ndarray_50_100 = np.random.randint(low=50, high=101, size=(6))
    print('\n<np.random.randint(low=50, high=101, size=(6))> returns a ndarray between 50-100\n',
        rand_int_ndarray_50_100 ,  rand_int_ndarray_50_100.shape, ',len =',len(rand_int_ndarray_50_100))                        
    # we use these ndarray to create matrix np.array([ndarray]):
    rand_int_vector = np.array([rand_int_ndarray_50_100]) # [] is used
    print('\nBut we used that ndarray to build a 1x6 random vector populated by random integers\n', rand_int_vector,
        rand_int_vector.shape, ',len =',len(rand_int_vector))
    print("**********************************************************************")

    # Generating and populating matrix using Random float Numbers
    random_floats_ndarray_0_and_1 = np.random.random([6])
    random_float_vector = np.array([random_floats_ndarray_0_and_1])
    
    print('\n a 1 x 6 random vector populated by six float numbers between 0 and 1:\n', random_float_vector,
                            random_float_vector.shape)
    print("**********************************************************************")

    # initialize a matrix of random numbers of a given shape with np.random.rand()
    # Generatig a random n x m matrix, another way
    d = np.random.rand(4, 2)
    # print(np.dot(d,d)) # ilegal operation  
    print("creating a 4 x 2 matrix:\n", d,'\nshape :', d.shape)
    print("creating a 2 x 4 matrix:\n", d.T,'\nshape :', d.T.shape)

    print("**********************************************************************")

    # Matrix multiplication
    M = np.random.rand(6, 6)
    #print(' M * M:\n',np.dot(M,M))
    #print(M.shape)
    #print("**********************************************************************")
    P = np.random.rand(6,6)
    #print('matrix P * P:\n', np.matmul(P, M))
    #print(P.shape)
    print("**********************************************************************")
    
    # We can get a ndarray list of each row or column of a matrix by:
    m_last_column = M[:, -1]
    print('Matrix M:\n', M)
    print(M.shape)
    print('The last column of matrix M is a list,a ndarray:\n', m_last_column, '\nshape:', m_last_column.shape)
    print("**********************************************************************")
    p_sec_col_step2 = P[:,1:2]
    print('\nMatrix P:\n', P)
    print(P.shape)
    print('The second column and 3rd columns of matrix P is a 2-D matrix not a ndarray:\n',
     p_sec_col_step2, '\nshape:', p_sec_col_step2.shape)
    print("**********************************************************************")

    # 2.4) Row vector
    # Takes a list of numbers and creates a 2D numpy array representing a column vector containing those numbers.
    my_list1 = [1,2,3,4,5,6,7]
    row_vector = np.array([my_list1]) # It is fucking important, put list into a [] here to have 2-D matrix
    print('\nRow Vector shape:', row_vector.shape,'\n', row_vector)
    print("**********************************************************************")

    # 2.5) Column vector
    # Takes a list of numbers and creates a 2D numpy array representing a column vector containing those numbers.
    column_vector = np.array([my_list1]).T
    print('\nColumn Vector shape:', column_vector.shape,'\n', column_vector)
    print("**********************************************************************")

    # 2.6) Euclidean length of column vector, ||column_vector||2 = np.sqrt(np.sum(np.square(column_vector)))
    # Takes a column vector and returns the vector's Euclidean length (or equivalently, its magnitude) 
    # as a scalar You may not use np.linalg.norm, and you may not use a loop.
    '''
    we can think of coding the Euclidean norm as a three step process:
    1. Square each item of the original vector (this can be done with 
        element-wise multiplication, col_v*col_v, or the built-in square function, np.square(col_v))
    2. Sum up all the elements in the new vector (this can be done with np.sum())
    3. Take the square root of the previous sum.
    '''
    print('first element of column_vector =', column_vector[0][0], type(column_vector[0][0]))
    print('last element of column_vector =', column_vector[-1][0])
    # But if do not use [][] format, you'll get a list not int
    # square function of column_vector, np.square(column_vector)
    square = np.square(column_vector)
    print('\nsquared column vector:\n',square)
    # sum ups all elements inside the squared vector, np.sum()
    sum_scalar = np.sum(square)
    print('summing of all elements will be a scalar:\n',sum_scalar)
    euclidean_length = np.sqrt(sum_scalar)
    print('Euclidean length of column_vector:\n', euclidean_length)
    print('Way2: can use this way, np.sum(col_v * col_v)**0.5\n', np.sum(column_vector * column_vector)**0.5)
    #print(np.sqrt((np.sum(np.square(column_vector)))))
    '''
    loop is not allowed in the problem
    sum = 0
    for ele in square:
        sum += ele[0]
    print(sum)    
    '''

    # 2.7) Normalize 
    # Takes a column vector and returns a unit vector in the same direction.
    # we derived the definition of a unit vector normal to the hyperplane specified by:
    #  Θ / ||Θ|| or -Θ / ||Θ||  
    # In this problem, our \thetaθ is our column vector column_vector,
    #  and our norm ||.|| is the length function we defined in the previous problem
    unit_vector1 = column_vector/euclidean_length
    unit_vector2 = -(column_vector/euclidean_length)
    print('\nso the unit vectors normal to the hyperplane specified by column_vector are:')
    print("unit vector1\n", unit_vector1)
    print("unit vector2\n", unit_vector2)
    print('column_vector\n', column_vector)
    # However, only one of these unit_vectors is in the same direction as column_vector
    # unit_vector1 is my normal vector to the hyperplane specified by column_vector
    print("unit_vector1 is my normal vector to the hyperplane specified by column_vector because is the same sign")
    print("**********************************************************************\n")

    # 2.8) Indexing
    # Write a procedure that takes a 2D array and returns the final column as a two dimensional array. 
    # You may not use a for loop.
    print('Matrix M\n',M) 
    (x, y) = M.shape
    print((x,y))
    # slicing over matrix M to get the final column as a list
    final_column_array = M[:, -1]
    print('the final column of matrix M as a list is\n', final_column_array, ', shape:',
        final_column_array.shape, type(final_column_array))
    # slicing over matrix M to get the final column as a two dimensional array.
    last_column_2D_matrix = M[:, -1:]
    print('last column as a 2D matrix\n', last_column_2D_matrix, ', shape:',
        last_column_2D_matrix.shape, type(last_column_2D_matrix))
    # creating a new 2-D array using final_col_array
    new_2D_array = np.array([final_column_array])
    print('\nThe new_2D_array\n',new_2D_array, ', shape:',new_2D_array.shape)
    print("**********************************************************************\n")

    # 2.9) Representing data
    # Alice has collected weight and height data of 3 people and has written it down below:
    '''
    Weight, height
    150, 5.8
    130, 5.5
    120, 5.3
    '''
    # 2.10) Now she wants to put this into a numpy array such that each row represents
    # one individual's height and weight in the order listed.
    #  Write code to set data equal to the appropriate numpy array:
    my_data = np.array([[150,5.8],[130,5.5],[120,5.3]])
    print('\nAlice data:\n', my_data, my_data.shape)
    nn = np.array([[1,1]]).T
    print('\n',nn, nn.shape)
    sum_row = np.matmul(my_data, nn)
    print('\n', sum_row, sum_row.shape)
    # Here my_data is a 3x2 matrix. We can multiply it by a 2x1 column vector,nn,
    #  of 1s to sum each row of my_data. This then gives us a 3x1 column vector
    #  representing sums of each person's height and weight.
    print("**********************************************************************\n")


    # Add a row to existing numpy 2d
    a = np.array([[1,2,3],[4,5,6]])
    print(a, a.shape)
    row = np.array([[7,8,9]])
    a = np.vstack([a, row])
    print(a, a.shape)
    # Add a column to existing numpy 2d
    col = np.array([[10,11,12]]).T # this array should be in same d as a
    a = np.hstack((a, col))
    print(a, a.shape)
    print("the maximum value in a is in index", np.argmax(a))
    print("**********************************************************************\n")


    
    




    


if __name__ == '__main__':
    main()    