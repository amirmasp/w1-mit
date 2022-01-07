# This is a Linear Learning Algorithem
# This is just a raw algorithem 
import sys   
import numpy as np
import hw01 as hw1 

def random_linear_class (D,k):
    '''
    We are about to writing a learning algorithem that takes training Data
    as input and create k hypotheses, collect them into a list and name it 
    the Hypothesis Class of a linear classifier,H. and returns the hypothesis that have
    had smallest training error between the other generated hypotheses.

    '''
    # inside D inhabit n Xs.
    # First we need to create the hypothesis class, H,
    #H = []
    print("We got data as:\n", D)
    print("\nk=",k)


def data_processing(data, k, labels):   
    scores = []
    # data is a 2_D numpy array
    (d , n) = data.shape
    m = k # We got he m, hyperparameter
    # We define Hypothesis class to be a d by k array (d rows, m columns) of random floats.
    # ths: a d by m array of floats representing m candidate θ's (each θ has dimension d by 1)
    # th0s: a 1 by m array of the corresponding m candidate 0θ 
    ths = np.random.rand(d, m)
    print('ths:\n',ths, ths.shape)
    ths0 = np.random.rand(1, m)
    print('ths0:\n', ths0, ths0.shape)
    # Loop through hypotheses
    for j in range(m):
        value_list = [] 
        Θ = ths[:,j:j+1] 
        print('\nvector Θ:\n',Θ, Θ.shape)
        print('offset Θ0:\n',ths0[:,j], ths0[:,j:j+1].shape)
        # loop through datapoint
        for i in range(n):
            data_point = data[:,i:i+1]
            # Now call for positive function to find out given hyperplane Θ, whether data_point is
            # in the positive side of h or negative side?
            p = hw1.ex1.positive(data_point, Θ, ths0[:,j]) # this returns +1 or -1 or 0
            #print('p\n', p[0,0]) 
            value_list.append(p[0,0]) # appending only value inside p to the list
        A =  hw1.ex1.rv(value_list)
        print('for hypothesis',j ,', A:\n',A, A.shape) 
        # A is a 1 by n array like labels 
        # compare A with labels to find score of a hypothesis
        B = A == labels
        print('labels:\n', labels, labels.shape)
        print('comparing A and labels, B:\n',B)
        print('score for hypothesis',j, 'is',np.sum([B]))
        print('appending to scores list\n')
        scores.append(np.sum([B]))
    print("scores\n", scores)
    # Now we have the scores list we can inds the hyperplane with the highest score on the data and labels.
    # In case of a tie, return the first hyperplane with the highest score, in the form of:
    # a tuple of a d by 1 array and an offset in the form of 1 by 1 array.
    max_score  = max(scores)
    print('highest score in list is', max_score)
    max_index = scores.index(max_score)
    print('index of smallest score is[',max_index,']')
    best_seperatorΘ = ths[:,max_index]
    best_seperatorΘ0 = ths0[:,max_index]
    print('\nWe found the best seperator Θ \n', best_seperatorΘ, best_seperatorΘ.shape)
    print('We found the best seperator Θ0 \n', best_seperatorΘ0, best_seperatorΘ0.shape)
    

    return scores





def main():
    # We define data to be a 3 by 5 array (three rows, five columns) of random floats.
    # data: a d by n array of floats (representing n data points in d dimensions)
    # labels: a 1 by n array of elements in (+1, -1), representing target labels 
    
    d = 3 # dimension of data and Hypothesis class
    n = 5 # n data point in d dimension in data
    m = 6 # hyperParameter is actually the number of hypotheses in Hypothesis class
    labels = hw1.ex1.rv([+1,-1,-1,+1,+1]) # label is a 1 by n array
    #data = np.random.rand(d, n)
    data = np.transpose(np.array([[1, 5,2], [1, 8,3], [2,3, 1], [1,-2,-1], [2, 1,-1]]))
    print('data:\n',data, data.shape)
    
    scores = data_processing(data, m , labels)
    #print('\ndatapoint are recived\n', datapoints)



if __name__ == '__main__':
    main()    