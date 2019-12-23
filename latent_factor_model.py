
"""
IR Assignment-3
Recommender System Using Latent Factor Model Technique
Authors: Dhruv Shah
        Vedant Goyal
        Ashish Gupta
"""

"""
All the necessary libraries and files are imported here such as numpy, os, operator etc.
"""

import numpy as np
import pandas as pd

def building_ratings_dictionary_wrt_movies_and_users():
	"""
     movies_file is a file that contains information about all 
       the movies present in the database
    
     ratings_temp is a file that contains information about all 
       the ratings given by all the users to all the movies
       present in the database
    
     users_temp is a file that contains information about all 
       the users present in the database
    
     movie_rate is a dictionary of dictionaries which is of the form
        {movie1:{user1:rating1,user2:rating2...},movie2:{...}...}
        where the keys are the all the movies of the database and the values are
        the dictionaries where the key is the user and his/her rating to that movie
        the user is the part of the values of that movie iff he has given rating to that 
        movie
    
     user_rate is a dictionary of dictionaries which is of the form
        {user1:{movie1:rating1,movie2:rating2...},user2:{...}...}
        where the keys are the all the users of the database and the values are
        the dictionaries where the key is the movie and the rating of that movie
        the movie is the part of the values of that user iff the user has given rating to that 
        movie
    
            All the three files are read  and we use the ratings_temp file to make the user_rate and the 
        movie_rate.These 2 dictionaries store the information required to implement the latent factor model
    
    """
file= open("movies.dat",'r')
movies_file=file.read()

file= open("ratings.dat",'r')
ratings_temp=file.read()

file= open("users.dat",'r')
users_temp=file.read()

ratings=ratings_temp.split("\n") 

movie_rate={}
user_rate={}
for i in range(0,len(ratings)-1):
    temp_list=ratings[i].split("::")
    movie_id=int(temp_list[1])
    user_id=int(temp_list[0])
    rating=int(temp_list[2])
    if movie_id in movie_rate:
        movie_rate[movie_id][user_id]=rating
    else:
        movie_rate[movie_id]={}
        movie_rate[movie_id][user_id]=rating
    
    if user_id in user_rate:
        user_rate[user_id][movie_id]=rating
    else:
        user_rate[user_id]={}
        user_rate[user_id][movie_id]=rating
        
def gradient_descent_to_implement_latent_factor_model():
    """
        latent_factors is number of latent factors are required
        eta is the learning rate
        lamda is the regularization parameter
        iter is the iteration number
    
         let (M)(Tra)=transpose of matrix M
         let sigma(par)=sum of all the elements around that parameter
         let A be the given ratings matrix.We have to split the matrix into 2
         matrices Q_matrix and P_matrix
         
         so that A=(Q_matrix)*((P_matrix)(Tra))
         
    
         Since we are applying gradient descent we have to 
             calculate and delta_Q and delta_P so that
         
         Q_matrix(time=t+1)=Q_matrix(time=t)-eta*delta_Q
         P_matrix(time=t+1)=P_matrix(time=t)-eta*delta_P
         
         where,
         
            delta_Q=sigma(user)(-2*((movie_rate[movie][user]-np.dot(Q_matrix[movie],P_matrix[user]))*P_matrix[user][latent_factor]))+2*lamda*P_matrix[movie][latent_factor]
            for a particular movie and a latent_factor
            
            delta_P=sigma(movie)(-2*((user_rate[user][movie]-np.dot(P_matrix[user],Q_matrix[movie]))*Q_matrix[movie][latent_factor]))+2*lamda*P_matrix[user][latent_factor]
    """
    

latent_factor=4
Q_matrix=np.ones( (3973,latent_factor) ) 
P_matrix=np.ones( (6041,latent_factor) ) 

eta=0.00002
lamda=2
delta_Q=np.zeros( (3973,latent_factor) ) 
delta_P=np.zeros( (6041,latent_factor) )

for iter in range(0,200):
    
    for row in range(0,len(Q_matrix)):
        for col in range(0,len(Q_matrix[0])):
            if row in movie_rate:
                sum=0
                for user in movie_rate[row]:
                    sum+=(movie_rate[row][user]-np.dot(Q_matrix[row],P_matrix[user]))*P_matrix[user][col]
                sum=sum*(-2)
                sum+=2*lamda*Q_matrix[row][col]
                delta_Q[row][col]=sum
    
    for row in range(0,len(P_matrix)):
        for col in range(0,len(P_matrix[0])):
            if row in user_rate:
                sum=0
                for movie in user_rate[row]:
                    sum+=(user_rate[row][movie]-np.dot(P_matrix[row],Q_matrix[movie]))*Q_matrix[movie][col]
                sum=sum*(-2)
                sum+=2*lamda*P_matrix[row][col]
                delta_P[row][col]=sum
                
    P_matrix-=eta*delta_P
    Q_matrix-=eta*delta_Q
    #print(P_matrix)
    #print(Q_matrix)
    
    #temp=np.transpose(P_matrix)
    error=0
    
    for i in range(1,6041):
        for movie in user_rate[i]:
            error+= (user_rate[i][movie]-np.dot(Q_matrix[movie],P_matrix[i]))**2
    print(iter)    
    print(error)
    print("\n")
    
def predict_and_calculate_MAE_and_RMSE():
    """
        predict[user][movie]= np.dot(Q_matrix[movie],P_matrix[user])
        
        RMSE=(sse/count)**0.5
        
        MAE=abs_error/count
        
        sse=((sigma(user,movie))(predict[user][movie]-user_rate[user][movie]))**2
    
        count=total number of ratings
        
        abs_error=(sigma(1 to count))abs((predict[user][movie]-user_rate[user][movie]))
    """
    
    
    
mean=0     
count=0
for i in range(1,6041):
    for movie in user_rate[i]:
        mean+=user_rate[i][movie]
        count+=1
mean/=count

sse=0
sst=0
maxi=0
mini=3
abs_error=0
#count=0
for i in range(1,6041):
    for movie in user_rate[i]:
        predict= np.dot(Q_matrix[movie],P_matrix[i])
        maxi=max(maxi,predict)
        mini=min(mini,predict)
        sse+=(predict-user_rate[i][movie])**2
        abs_error+=abs(predict-user_rate[i][movie])
       # count+=1
        sst+=(mean-user_rate[i][movie])**2
        
r2_score=1-(sse/sst)               

mae=abs_error/count

rmse=(sse/count)**0.5
    