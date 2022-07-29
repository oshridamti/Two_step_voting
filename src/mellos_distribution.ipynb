##Include##
import random
import numpy as np
import math
import statistics
import copy

def index_to_insert(denominator, i , sigma):
  random_num = random.random() 
  for j in range(1,i+1):
    Pij = pow(sigma, i-j)/denominator ## sigma^i-j/1+sigma+ ... + sigma ^ i-1
    ##print("P",i,j," = ",Pij)
    random_num = random_num - Pij     ## split 0-1 to intervals of 0-Pi1,Pi1-Pi2,....,Pii-1-Pii 
    if (random_num <=0) or (i==j):    ## check if random num is in interval Pij-1-Pij for defintion Pi0 = 0  
      return j-1                      ## 1-i but array index is from 0-i-1                     

def create_n_rannking_from_truthful_ranking_and_sigma(truthful_ranking, sigma=0.8):
  sampled_rank = []
  denominator = 1
  for (ranked_voter,i) in zip(truthful_ranking,range(1,len(truthful_ranking)+1)):
    index = index_to_insert(denominator, i, sigma)
    sampled_rank.insert(index, ranked_voter)   ## insert person truth rank number i to voter ranking in index
    denominator = denominator + pow(sigma,i)   ## Update denominator for next round denominator = 1+sigma+ ... + sigma ^ i-1
  return sampled_rank
