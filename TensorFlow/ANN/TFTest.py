'''
TFTest.py
This file contains unit tests for the TFANN module.
'''
import numpy as np
import tensorflow as tf
from TFANN import ANN, ANNC, ANNR, MLPB, MLPMC, RFSANNR
import traceback
import time
    
def RunTests(T):
    f = True
    for i, Ti in enumerate(T):
        t1 = time.time()
        try:
            rv = Ti()
        except Exception as e:
            traceback.print_exc()
            rv = False
        t2 = time.time()
        ANN.Reset()
        f = rv and f
        print('{:s} {:3d}:\t{:s}\t({:7.4f}s)'.format('Test ', i + 1, 'PASSED' if rv else 'FAILED', t2 - t1))
    if f:
        print('All tests passed!')
    else:
        print('Warning! At least one test failed!')
    
def T1():
    '''
    Tests basic functionality of ANNR
    '''
    A = np.random.rand(32, 4)
    Y = np.random.rand(32, 1)
    a = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 'mlpr1')
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T2():
    '''
    Tests basic functionality of ANNC
    '''
    A = np.random.rand(32, 4)
    Y = np.array((16 * [1]) + (16 * [0]))
    a = ANNC([4], [('F', 4), ('AF', 'tanh'), ('F', 2)], maxIter = 16, name = 'mlpc2')
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True

def T3():
    '''
    Tests basic functionality of MLPB
    '''
    A = np.random.randint(0, 2, size = (32, 5))
    Y = np.random.randint(0, 2, size = (32, 2))
    a = MLPB([5], [('F', 4), ('AF', 'tanh'), ('F', 2), ('AF', 'tanh'), ('F', 2)], maxIter = 16, name = 'mlpb1')
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T4():
    '''
    Tests basic functionality of RFSMLPB
    '''
    A = np.random.randint(0, 2, size = (32, 6))
    Y = np.random.randint(0, 2, size = (32, 4, 6))
    a = RFSMLPB([6, 6, 6, 6], maxIter = 12, name = 'rfsmlpb1')
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T5():
    '''
    Tests basic functionality of MLPMC
    '''
    A = np.random.rand(33, 5)
    Y = np.tile(['y', 'n', 'm'], 55).reshape(33, 5)
    a = MLPMC([5], 5 * [[('F', 4), ('AF', 'tanh'), ('F', 3)]], maxIter = 12, name = 'mlpmc1')
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T6():
    '''
    Tests basic functionality of CNNC
    '''
    A = np.random.rand(32, 9, 9, 3)
    Y = np.array((16 * [1]) + (16 * [0]))
    ws = [('C', [3, 3, 3, 4], [1, 1, 1, 1]), ('AF', 'relu'), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 16), ('AF', 'relu'), ('F', 2)]
    a = ANNC([9, 9, 3], ws, maxIter = 12, name = "cnnc1")
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T7():
    '''
    Tests basic functionality of CNNC
    '''
    A = np.random.rand(32, 9, 9, 3)
    Y = np.random.rand(32, 1)
    ws = [('C', [3, 3, 3, 4], [1, 1, 1, 1]), ('AF', 'relu'), ('P', [1, 4, 4, 1], [1, 2, 2, 1]), ('F', 16), ('AF', 'tanh'), ('F', 1)]
    a = ANNR([9, 9, 3], ws, maxIter = 12, name = "cnnr1")
    a.fit(A, Y)
    S = a.score(A, Y)
    if np.isnan(S):
        return False
    YH = a.predict(A)
    if Y.shape != YH.shape:
        return False
    return True
    
def T8():
    '''
    Tests if multiple ANNRs can be created without affecting each other
    '''
    A = np.random.rand(32, 4)
    Y = (A.sum(axis = 1) ** 2).reshape(-1, 1)
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16)
    m1.fit(A, Y)
    R1 = m1.GetWeightMatrix(0)
    m2 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16)
    m2.fit(A, Y)
    R2 = m1.GetWeightMatrix(0)
    if (R1 != R2).any():
        return False
    return True
    
def T9():
    '''
    Tests if multiple ANNRs can be created without affecting each other
    '''
    A = np.random.rand(32, 4)
    Y = (A.sum(axis = 1) ** 2).reshape(-1, 1)
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16)
    m1.fit(A, Y)
    s1 = m1.score(A, Y)
    m2 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16)
    m2.fit(A, Y)
    s2 = m1.score(A, Y)
    if s1 != s2:
        return False
    return True
    
def T10():
    '''
    Tests if multiple ANNCs can be created without affecting each other
    '''
    A = np.random.rand(32, 4)
    Y = np.array((16 * [1]) + (16 * [0]))
    m1 = ANNC([4], [('F', 4), ('AF', 'tanh'), ('F', 2)], maxIter = 16, name = 'mlpc2')
    m1.fit(A, Y)
    s1 = m1.score(A, Y)
    m2 = ANNC([4], [('F', 4), ('AF', 'tanh'), ('F', 2)], maxIter = 16, name = 'mlpc3')
    m2.fit(A, Y)
    s2 = m1.score(A, Y)
    if s1 != s2:
        return False
    return True
    
def T11():
    '''
    The XOR problem
    '''
    A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0], dtype = np.int)
    m1 = ANNC([2], [('F', 2), ('AF', 'tanh'), ('F', 2)], batchSize = 3, learnRate = 1e-5, maxIter = 4096, name = 'mlpc2', tol = 0.4)
    m1.fit(A, Y)
    if m1.score(A, Y) < 0.5:
        return False
    return True
    
def T12():
    '''
    Tests saving a model to file
    '''
    A = np.random.rand(32, 4)
    Y = (A.sum(axis = 1) ** 2).reshape(-1, 1)
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 't12ann1')
    m1.fit(A, Y)
    m1.SaveModel('./t12ann1')
    return True

def T13():
    '''
    Tests restoring a model from file
    '''
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 't12ann1')
    rv = m1.RestoreModel('./', 't12ann1')
    return rv
    
def T14():
    '''
    Tests saving and restore a model
    '''
    A = np.random.rand(32, 4)
    Y = (A.sum(axis = 1) ** 2).reshape(-1, 1)
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 't12ann1')
    m1.fit(A, Y)
    m1.SaveModel('./t12ann1')
    R1 = m1.GetWeightMatrix(0)
    ANN.Reset()
    m1 = ANNR([4], [('F', 4), ('AF', 'tanh'), ('F', 1)], maxIter = 16, name = 't12ann2')
    m1.RestoreModel('./', 't12ann1')
    R2 = m1.GetWeightMatrix(0)
    if (R1 != R2).any():
        return False
    return True
    
def T15():
    '''
    Tests setting weight matrix and bias vectors
    '''
    m1 = ANNR([4], [('F', 4), ('F', 2)], maxIter = 16, name = 't12ann1')
    m1.SetWeightMatrix(0, np.zeros((4, 4)))
    m1.SetBias(0, np.zeros(4))
    m1.SetWeightMatrix(1, np.zeros((4, 2)))
    m1.SetBias(1, np.zeros(2))
    YH = m1.predict(np.random.rand(16, 4))
    if (YH != 0).any():
        return False
    m1.SetWeightMatrix(0, np.ones((4, 4)))
    m1.SetBias(0, np.ones(4))
    m1.SetWeightMatrix(1, np.ones((4, 2)))
    m1.SetBias(1, np.ones(2))
    YH = m1.predict(np.ones((16, 4)))
    if np.abs(YH - 21).max() > 1e-5:
        print(np.abs(YH - 21).max())
        return False
    return True
    
if __name__ == "__main__":
    RunTests([T1, T2, T3, T5, T6, T7, T8, T9, T10, T11, T12, T13, T15])
