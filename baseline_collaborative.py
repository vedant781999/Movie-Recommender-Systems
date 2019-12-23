
import time

import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank
from numpy import linalg as la

movies_dataset = open("ml-1m/movies.dat",'r')
rating_dataset = open("ml-1m/ratings.dat",'r')
user_dataset = open("ml-1m/users.dat",'r')

rating_data = rating_dataset.readlines()
movies_data = movies_dataset.readlines()
user_data = user_dataset.readlines()

rating_list = [[] for i in range(len(rating_data))]
for i in range(len(rating_data)):
    ratings = rating_data[i].split('::')
    rating_list[i].extend(ratings)
    
user_list = [[] for i in range(len(user_data))]
for i in range(len(user_data)):
    users = user_data[i].split('::')
    user_list[i].extend(users)
    
movie_list = [[] for i in range(len(movies_data))]
for i in range(len(movies_data)):
    movies = movies_data[i].split('::')
    genre_list = movies[2].split('|')
    movie_list[i].extend(movies[0:2])
    movie_list[i].extend(genre_list)
    
A = np.zeros([len(user_list)+1,3953],dtype = int)
movie_length = 3953
user_length = 6041

for i in range(len(rating_list)):
    A[int(rating_list[i][0])][int(rating_list[i][1])] = int(rating_list[i][2])
    
    
item_user_matrix = A.T

similarity_matrix  = np.zeros((3953,3953))
similarity_matrix.fill(-2)

item_user_matrix_row_means = np.zeros((3953,1))
 
for i in range(movie_length):
    item_user_matrix_row_means[i] = np.mean(item_user_matrix[i])
          
print( time.asctime( time.localtime(time.time()) ))    
for i in range(movie_length):
    for j in range(i+1, movie_length):
        temp = la.norm(item_user_matrix[i] - item_user_matrix_row_means[i])*la.norm(item_user_matrix[j] - item_user_matrix_row_means[j])
        similarity_matrix[i][j] = np.dot(item_user_matrix[i] - item_user_matrix_row_means[i],item_user_matrix[j] - item_user_matrix_row_means[j] )
        if temp!=0:
            similarity_matrix[i][j]/= temp
        else:
            temp=1
            similarity_matrix[i][j]/= temp
        similarity_matrix[j][i] = similarity_matrix[i][j] 
        
print( time.asctime( time.localtime(time.time()) ))

baseline_matrix = np.zeros((movie_length, user_length))
movie_mean_matrix = np.zeros((movie_length, 1))
user_mean_matrix = np.zeros((user_length, 1))


total_count=0
for i in range(user_length):
    sum=0
    count=0
    for j in range(movie_length):
        if item_user_matrix[j][i] != 0:
            count+=1
        sum+=(item_user_matrix[j][i])
    total_count+=count
    if(count!=0):    
        sum/=count 
    user_mean_matrix[i] = sum
   
for j in range(movie_length):
    sum=0
    count=0
    for i in range(user_length):
        if item_user_matrix[j][i] != 0:
            count+=1
        sum+=(item_user_matrix[j][i])
    if(count!=0):
        sum/=count
    movie_mean_matrix[j]=sum
    
overall_mean = np.sum((item_user_matrix))/total_count

    
# =============================================================================
# for i in range(user_length):
#     user_mean_matrix[i] = np.mean((item_user_matrix[:,i]))
#     
# overall_mean = item_user_matrix.mean()
# =============================================================================

for i in range(movie_length):
    for j in range(user_length):
        baseline_matrix[i][j] = user_mean_matrix[j] + movie_mean_matrix[i] - overall_mean

#### Only for 1 prediction . top 3 movies similar to a movie
answer = similarity_matrix[1].argsort()[-3:][::-1]
answer_list = [[] for i in range(3)]
answer_list = np.zeros((3,6041))
for i in range(0,3):
    answer_list[i] = (np.array((item_user_matrix[answer[i]]))) 
####	
	
def constructing_predicted_matrix():
    """
    answer returns the similarity of a movie with the top 3 movies
    predicted_rating is a matrix containing predicted ratings
    """
predicted_rating= np.zeros((3953,6041))
count=0
for i in range(0, 3953):
    answer = similarity_matrix[i].argsort()[-3:][::-1]
    for j in range(0, 6041):
        sum=0
        sum1=0
        if item_user_matrix[i][j]!=0:
            for k in range(0,3):
                sum+=similarity_matrix[i][answer[k]]*(item_user_matrix[answer[k]][j] - baseline_matrix[answer[k]][j])
                sum1+=similarity_matrix[i][answer[k]]
                count+=1
				temp = sum/sum1
            predicted_rating[i][j]= temp + baseline_matrix[i][j]

def error_calculation():
    """
    Here, rmse and mae is calculated accordingly from the predicted matrix and the actual matrix.
    """
diff = item_user_matrix-predicted_rating
sq = diff**2
sum = np.sum(sq)
sum/=(count)
rmse = sum**0.5

abs_diff = np.absolute(diff)
mae = abs_diff/count

print(rmse, mae)	
