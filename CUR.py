import numpy as np
#import pandas as pd
import math
import random
from numpy.linalg import matrix_rank

def construction_of_input_matrix():
    """Loaded the ratings file and prepared the input matrix A(User*Movie)"""

ratings_file = open("ml-1m/"+"ratings.dat",'r')
ratings_data=ratings_file.readlines()
num_of_ratings=len(ratings_data)
user_len=6041
movie_len=3953
A=np.zeros((6041,3953),"int32")      # Input matrix
for i in range(0,num_of_ratings):
    line=ratings_data[i]
    line=line.split("::")
    A[int(line[0])-1][int(line[1])-1]=int(line[2])

def calculate_probablity_rows_cols():
    """Calculating the probablities of individual rows and columns
    user_prob array is for rows and movie_prob is for columns
    All probablities are divided by frobenius norm of the entire matrix"""
total_square_sum=0
user_prob=np.zeros((user_len,1))
movie_prob=np.zeros((movie_len,1))
for i in range(0,user_len):
    sum=0
    for j in range(0,movie_len):
        sum+=(A[i][j]**2)
        total_square_sum+=(A[i][j]**2)
    user_prob[i]=sum
   
for j in range(0,movie_len):
    sum=0
    for i in range(0,user_len):
        sum+=(A[i][j]**2)
    movie_prob[j]=sum
    
for i in range(0,user_len):
    user_prob[i]/=total_square_sum
    
for i in range(0,movie_len):
    movie_prob[i]/=total_square_sum
    
#-------------------
def cumulative_frequency():
    """Calculating the cumulative frequency of user_prob and movie_prob
    # for selecting rows and columns randomly accofrding to the corresponding probablities"""
cdf_user_prob=np.zeros((user_len,1))
sum=0
for i in range(0,user_len):
    sum+=user_prob[i]
    cdf_user_prob[i]+=sum
    
    
cdf_movie_prob=np.zeros((movie_len,1))
sum=0
for i in range(0,movie_len):
    sum+=movie_prob[i]
    cdf_movie_prob[i]+=sum

#sum=total_square_sum
def R_matrix():
    """finding random rows
    #Making the R matrix by selecting rows randomly and according to their probablity"""
r=2000
R=np.zeros((r,movie_len),)
selected_rows_index=np.zeros((r,1),"int32")
for i in range(0,r):
    rand=random.random()
    print(rand)
    selected_row=-1
    for j in range(0,user_len):
        if(rand<cdf_user_prob[j][0]):
            selected_row=j
            selected_rows_index[i]=j
            print(selected_row)
            break
    R[i]=A[selected_row]/((r*user_prob[selected_row])**0.5)

def C_matrix():
    """Making the C matrix by selecting columns randomly and according to their probablity"""


r=2000
C=np.zeros((user_len,r),)
selected_cols_index=np.zeros((r,1),"int32")
for i in range(0,r):
    rand=random.random()
    #print(rand)
    selected_col=-1
    for j in range(0,movie_len):
        if(rand<cdf_movie_prob[j][0]):
            selected_col=j
            selected_cols_index[i]=[j]
            #print(selected_col)
            break
    for j in range(0,user_len):
        C[j][i]=A[j][selected_col]/((r*movie_prob[selected_col])**0.5)

def U_matrix():        
    """Making the temporary U matrix"""
        
U=np.zeros((r,r))
for i in range(0,r):
    for j in range(0,r):
        #print(selected_rows_index[i][0],selected_cols_index[j][0])
        U[i][j]=A[selected_rows_index[i][0]][selected_cols_index[j][0]]

def apply_SVD():
    """Applying SVD on U matrix and pseudo inverse on S"""

X, S_temp, YT = np.linalg.svd(U, full_matrices=True)
S=np.zeros((r,r))
for i in range(0,r):
    S[i][i]=S_temp[i]

     if S[i][i]!=0:
         S[i][i]=1/S[i][i]

        
        
Y = YT.T
#S = np.linalg.pinv(S)
#S=np.dot(S,S)
S=np.dot(S,S)
XT=X.T
new_U=np.dot(Y,S)
new_U=np.dot(new_U,XT)

#Calculating the predicted CUR matrix

predicted_CUR=np.dot(C,new_U)
predicted_CUR=np.dot(predicted_CUR,R)

def ninty_percent_CUR():
    """Calculating 90% CUR by applying 90% SVD on the U matrix"""
X, S_temp, YT = np.linalg.svd(U, full_matrices=True)
total_sum_diag_elements=np.sum(S_temp)
sum_90=total_sum_diag_elements*0.9
sum=0
br_90=0
for i in range(0,r):
    sum+=S_temp[i]
    if(sum>=sum_90):
        br_90=i+1
        break
S=np.zeros((br_90,br_90))
for i in range(0,br_90):
    S[i][i]=S_temp[i]
    if S[i][i]!=0:
        S[i][i]=1/S[i][i]
X=X[:,:br_90]
YT=YT[:br_90,:]
S=np.dot(S,S)
Y = YT.T
XT=X.T
new_U=np.dot(Y,S)
new_U=np.dot(new_U,XT)
predicted_CUR=np.dot(C,new_U)
predicted_CUR=np.dot(predicted_CUR,R)

def RMS_caalculation():
    """Calculating rms"""

subs=0
rms_subs=0
cnt=0
for i in range(0,r):
    for j in range(0,r):
        #if A[selected_rows_index[i][0]][selected_cols_index[j][0]]!=0:
        z=abs(predicted_CUR[selected_rows_index[i][0]][selected_cols_index[j][0]]-A[selected_rows_index[i][0]][selected_cols_index[j][0]])
        subs+=z
        rms_subs+=(z**2)
        cnt+=1
subs/=cnt
rms_subs/=cnt
rms_subs=rms_subs**0.5
rms_subs 

