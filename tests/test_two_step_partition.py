phi = [0.75, 0.8, 0.85, 0.9]
num_of_voters = [20, 30, 40, 50, 60, 70, 80]
winners_percentage = [5, 10, 15, 20, 25, 30, 35]
num_of_reviews_per_candidate=[5, 7, 9, 11]
split_step=[0, 0.25, 0.5, 0.75]
first_step_winners=[0, 0.25, 0.5, 0.75]
first_step_losers=[0, 0.25, 0.5, 0.75]
prob_for_not_rank = [0.1, 0.3, 0.5, 0.7]


num_of_iterations = 1000
num_of_tests = 50

color = ""
success_params = []
failed_params = []

num_of_new_method_win = 0
num_of_old_method_win = 0
num_of_no_winner = 0

for test_num in range(1, num_of_tests+1):
  
  best_avg_dis = 0
  best_avg_avg = 0
  best_avg_avg_Dis = 0

  all_vote_avg_dis = 0
  all_vote_avg_avg = 0
  all_vote_avg_avg_Dis = 0

  new_avg_dis = 0
  new_avg_avg = 0
  new_avg_avg_Dis = 0

  old_avg_dis = 0
  old_avg_avg = 0
  old_avg_avg_Dis = 0

  phi_val = random.choice(phi)
  num_of_voters_val = random.choice(num_of_voters)
  winners_percentage_val = random.choice(winners_percentage)
  num_of_reviews_per_candidate_val = random.choice(num_of_reviews_per_candidate)
  split_step_val = random.choice(split_step)
  first_step_winners_val = random.choice(first_step_winners)
  first_step_losers_val = random.choice(first_step_losers)
  prob_val = random.choice(prob_for_not_rank)


  for i in range(num_of_iterations):
    #print("\nIteartion number ", i, "\n")
    partition_class_test = partition(2,num_of_voters=num_of_voters_val,truthful_ranking=np.array(range(1,num_of_voters_val+1)),sigma=phi_val,winners_percentage=winners_percentage_val,num_of_reviews_per_candidate=num_of_reviews_per_candidate_val,split_step=split_step_val,first_step_winners=first_step_winners_val,first_step_losers=first_step_losers_val,prob_for_not_rank=prob_val,new_method=True)
    best_method_winners, all_votes, new_method_winners, old_method_winners = partition_class_test.train_vote()
    #print("new_method_winners: ", new_method_winners, " best_method_winners: ", best_method_winners, "all_votes: ", all_votes)

    best_avg, best_dis, best_avg_Dis = partition_class_test.accuracy(best_method_winners).values()
    best_avg_dis+= best_dis
    best_avg_avg+= best_avg
    best_avg_avg_Dis += best_avg_Dis

    all_vote_avg, all_vote_dis, all_vote_avg_Dis = partition_class_test.accuracy(all_votes).values()
    all_vote_avg_dis+= all_vote_dis
    all_vote_avg_avg+= all_vote_avg
    all_vote_avg_avg_Dis += all_vote_avg_Dis

    new_avg, new_dis, new_avg_Dis = partition_class_test.accuracy(new_method_winners).values()
    new_avg_dis+= new_dis
    new_avg_avg+= new_avg
    new_avg_avg_Dis += new_avg_Dis
    #print("dis ", new_dis, " avg ", new_avg, " avgDis ", new_avg_Dis)

    old_avg, old_dis, old_avg_Dis = partition_class_test.accuracy(old_method_winners).values()
    old_avg_dis+= old_dis
    old_avg_avg+= old_avg
    old_avg_avg_Dis += old_avg_Dis
  if (new_avg_dis < old_avg_dis):
    success_params.append({'sigma:': phi_val, 'num_of_voters: ' : num_of_voters_val, 'winners_percentage: ' : winners_percentage_val, 'num_of_reviews_per_candidate: ' : num_of_reviews_per_candidate_val, 'split_step: ' : split_step_val ,'first_step_winners: ' : first_step_winners_val, 'first_step_losers: ' : first_step_losers_val, 'prob_val: ' : prob_val })
    num_of_new_method_win += 1
    color = '\033[92m'
  elif (new_avg_dis == old_avg_dis):
    num_of_no_winner+=1
    color = '\033[93m'
  else:
    num_of_old_method_win += 1
    color = '\033[91m'
    failed_params.append({'sigma:': phi_val, 'num_of_voters: ' : num_of_voters_val, 'winners_percentage: ' : winners_percentage_val, 'num_of_reviews_per_candidate: ' : num_of_reviews_per_candidate_val, 'split_step: ' : split_step_val ,'first_step_winners: ' : first_step_winners_val, 'first_step_losers: ' : first_step_losers_val})

  print("\n")
  print(color+"Test number: ", test_num)
  #print("\nSummary for phi: ", phi_val, " and Num of iterations: ", num_of_iterations, " num of winners: ", partition_class_test.num_of_winners, " num of first votes: ", partition_class_test.num_of_first_votes)
  partition_class_test.print_test_summary()
  print("Best method:")
  print("Avg dis: ", best_avg_dis/num_of_iterations)
  print("Avg avg: ", best_avg_avg/num_of_iterations)
  print("Avg avgDis: ", best_avg_avg_Dis/num_of_iterations)

  print("\nall Vote method:")
  print("Avg dis: ", all_vote_avg_dis/num_of_iterations)
  print("Avg avg: ", all_vote_avg_avg/num_of_iterations)
  print("Avg avgDis: ", all_vote_avg_avg_Dis/num_of_iterations)

  print("\nNew method:")
  print("Avg dis: ", new_avg_dis/num_of_iterations)
  print("Avg avg: ", new_avg_avg/num_of_iterations)
  print("Avg avgDis: ", new_avg_avg_Dis/num_of_iterations)

  print("\nOld method:")
  print("Avg dis: ", old_avg_dis/num_of_iterations)
  print("Avg avg: ", old_avg_avg/num_of_iterations)
  print("Avg avgDis: ", old_avg_avg_Dis/num_of_iterations)

  

print('\033[0m' + "Number of time new method was better: ", num_of_new_method_win, " number of time old method was better: ", num_of_old_method_win, " number of time of same score: ", num_of_no_winner)
print(success_params)
print("/n/n/n/n")
print(failed_params)
