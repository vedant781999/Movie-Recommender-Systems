"""
Singular Value Decomposition(SVD)
SVD Model is applied here to predict ratings of movies to users
"""

import math
import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank

def original_matrix_construction():
  """
  Here, the file ratings.dat contains the user ratings for respective movies.
  These ratings are further stored in a matrix A which is a user-movie matrix.
  A is a very sparse matrix.
  """

rating_dataset = open("ratings.dat",'r')
rating_data = rating_dataset.readlines()
rating_list = [[] for i in range(len(rating_data))]

for i in range(len(rating_data)):
    ratings = rating_data[i].split('::')
    rating_list[i].extend(ratings)

A = np.zeros([6041,3953],dtype = int)

for i in range(len(rating_list)):
    A[int(rating_list[i][0])][int(rating_list[i][1])] = int(rating_list[i][2])


def user_rank_matrix_construction():
  """
  U is a user - rank matrix. 
  U is calculated from AAT.
  sorted_eigen_vector contains the eigen_vectors which are the columns of the U matrix.
  sorted_eigen_value contains the eigen_values(lambda) for each eigen_vector in a descending sequence.
  """
transposed_A  = A.T
AAT = np.dot(A,transposed_A)

eigen_value, eigen_vector = np.linalg.eig(AAT)

sorted_indices = eigen_value.argsort()[::-1][:eigen_value.shape[0]]

sorted_eigen_value = eigen_value[sorted_indices]

sorted_eigen_vector = eigen_vector[:,sorted_indices]

U = np.zeros((eigen_vector.shape[0],eigen_vector.shape[1]))

for i in range(eigen_vector.shape[1]):
  U[:,i] = sorted_eigen_vector[:,i]


def sigma_matrix_construction():

  """
  S is a rank-rank diagonal matrix whose diagonals are the sqrt of the eigenvalues (lambda ).
  The diagonal values are in descending order .
"""
rank = 3663
sliced_U = U[:,0:3663]
sqrt_S = np.sqrt(sorted_eigen_value)
I = np.eye(6041,3953)
for i in range(3953):
  I[i][i] *= sqrt_S[i]
S = I


def rank_movie_matrix_construction():

  """
  VT is a rank - movie matrix. 
  V is calculated from ATA.
  sorted_eigen_vector contains the eigen_vectors which are the columns of the V matrix.
  sorted_eigen_value contains the eigen_values(lambda) for each eigen_vector in a descending sequence.
  """
ATA = np.dot(transposed_A,A)

eigen_value_new, eigen_vector_new = np.linalg.eig(ATA)

sorted_indices_new = eigen_value_new.argsort()[::-1][:eigen_value.shape[0]]
sorted_eigen_value_new = eigen_value_new[sorted_indices_new]
sorted_eigen_vector_new = eigen_vector_new[:,sorted_indices_new]

V = np.zeros((eigen_vector_new.shape[0],eigen_vector_new.shape[1]))

for i in range(eigen_vector_new.shape[1]):
  V[:,i] = sorted_eigen_vector_new[:,i]

sliced_V = V[:,0:3663]

VT = V.T


def svd_error_calculation():
  """
  predicted_A is the final predicted matrix derived from applying SVD to the original matrix.
  Here, rmse and mae is calculated accordingly from the predicted matrix and the actual matrix.
  """

product = np.dot(U, S)

predicted_A = np.dot(product, VT)

error=0
sum=0
for i in range(6041):
  for j in range(3953):
    sq = (A[i][j]-predicted_A[i][j])**2
    sum+= abs(A[i][j]-predicted_A[i][j])
    error+=sq
error/=(6041*3953)
mae = sum/(6041*3953)    
rmse = math.sqrt(error)

print(rmse,mae)


def svd_90_percent_energy():

  """
  Here , we first calculate the total energy which is the sum of diagonal values of the matrix S.
  Now, starting from the last diagonal element we remove that element and calculate the energy.
  We iterate over this till we get the ratio of energy /total energy < 90% .
  We now delete equal number of columns (i.e., corresponding eigenvectors) from U and V. (i.e., delete rows from VT)
  """

total_energy=0
for i in range(3953):
  total_energy+=S[i][i]

total_energy

energy=total_energy
count=0
for i in range(3953):
  count+=1
  energy-=S[3952-i][3952-i]
  if energy/total_energy < 0.9:
    count-=1
    energy+=S[3952-i][3952-i]
    break

rank = 3953-count
sliced_U = U[:,0:rank]

sqrt_S = np.sqrt(sorted_eigen_value)
I = np.eye(rank,rank)
for i in range(rank):
  I[i][i] *= sqrt_S[i]
S = I

sliced_V = V[:,0:rank]

sliced_VT = sliced_V.T


def error_90_percent_energy_calculation():

  """
  Now, we check error on the SVD model with 90% energy.
  """

product = np.dot(sliced_U, S)
predicted_A = np.dot(product, sliced_VT)

error=0
sum=0
for i in range(6041):
  for j in range(3953):
    sq = (A[i][j]-predicted_A[i][j])**2
    sum+= abs(A[i][j]-predicted_A[i][j])
    error+=sq
error/=(6041*3953)
mae_90_percent = sum/(6041*3953)    
rmse_90_percent = math.sqrt(error)

print(rmse_90_percent,mae_90_percent)