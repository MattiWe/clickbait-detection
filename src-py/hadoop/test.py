import scipy.sparse
import numpy as np

x_train = scipy.sparse.csc_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.asarray([1, 0, 0])

x_train = scipy.sparse.load_npz('x_train.npz')
x_test = scipy.sparse.load_npz('x_test.npz')


# x_test = scipy.sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

y_train = np.load('y_train.npz')['data']
y_test = np.load('y_test.npz')['data']

print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)
# print(y_train)
# print(x_train)

# x_train = scipy.sparse.vstack((x_train, x_train[1]))
'''print(m)
print("\n")
# remove rows
print(m[[0, 2]])

print("\n")
# remove columns
print(m[:, [0, 2]])
'''
