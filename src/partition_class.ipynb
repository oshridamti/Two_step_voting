def get_winners_percentage_from_voter_sampled_rank(voter_sampled_rank,num_of_winners):
  #num_of_winners = (len(voter_sampled_rank)*winners_percentage)//100
  return voter_sampled_rank[:num_of_winners]

class partition:
  def __init__(self, num_of_partitions, num_of_voters, truthful_ranking=[],sigma=0,winners_percentage=3,num_of_reviews_per_candidate=7,split_step=0.5,first_step_winners=0.5,first_step_losers=0.7,prob_for_not_rank=0.3,new_method=False):
    self.winners_percentage = winners_percentage ## init winner percentage
    self.sigma = sigma  ## init sigma 
    self.num_of_partitions = num_of_partitions ## init num of_partitions
    self.num_of_voters = num_of_voters ## init num of voters
    self.voters_sampled_rank_by_partition = []  ## will use to keep for each partition each voter ranking
    self.num_of_reviews_per_candidate = num_of_reviews_per_candidate
    if not len(truthful_ranking):
      truthful_ranking = np.array(range(1, 201))
      np.random.shuffle(truthful_ranking)
    self.truthful_ranking = truthful_ranking
    self.voter_sampled_ranks = [create_n_rannking_from_truthful_ranking_and_sigma(truthful_ranking=np.delete(truthful_ranking, np.argwhere(truthful_ranking == voter)), sigma=sigma) for voter in range(1,num_of_voters+1)] ## Init ranks for each voter
    #self.voter_sampled_ranks = [create_n_rannking_from_truthful_ranking_and_sigma(truthful_ranking=np.delete(truthful_ranking, np.argwhere(truthful_ranking == voter)), sigma=sigma) for voter in range(1,num_of_voters+1)] ## Init ranks for each voter
    self.partition_number = num_of_voters//num_of_partitions
    modulo_of_winners = (num_of_voters*winners_percentage//100) % num_of_partitions
    self.num_of_winners = max(num_of_partitions,num_of_voters*winners_percentage//100 - modulo_of_winners)
    self.num_of_winners_per_partition = self.num_of_winners//num_of_partitions
    self.new_method = new_method
    self.split_step = split_step
    self.first_step_winners= first_step_winners
    self.first_step_losers = first_step_losers
    
    self.best_method = get_winners_percentage_from_voter_sampled_rank(truthful_ranking,self.num_of_winners)
    self.Voter_partition_i = [] ##voters partitions for debug
    self.iteration_i_unique = [] ## for more information
    self.iteration_i_counts = [] ## for more information
    self.itreration_train_vote_x_times = [] ## for more information
    self.partition_votes_to_candidates = []

    self.num_of_first_votes = max(1,math.ceil(self.num_of_reviews_per_candidate*self.split_step))
    self.num_of_first_votes_winners = math.ceil(self.num_of_winners*self.first_step_winners)
    self.num_of_first_votes_losers = math.ceil(self.num_of_winners*self.first_step_losers)
    self.num_of_first_votes_winners_per_partition = self.num_of_first_votes_winners//self.num_of_partitions
    self.num_of_first_votes_losers_per_partition = self.num_of_first_votes_losers//self.num_of_partitions
    self.num_of_second_votes = self.num_of_reviews_per_candidate - self.num_of_first_votes
    self.prob_for_not_rank = prob_for_not_rank



  def get_partition_winners(self,partition_votes,keep_log=False,draw_method="random"):
    unique, counts = np.unique(partition_votes, return_counts=True)  
    num_of_winners = len(partition_votes[0]) ##each voter give num of winners votes so num_of_winners == num of each voter

    if debug:
      print("Uniqe : " , unique)
      print("Counts : " ,counts)
      print("Sum of Counts : ", sum(counts))
      print("Choose : " , [x for _, x in sorted(zip(counts, unique), reverse=True)][:num_of_winners])
    
    if keep_log:
      self.iteration_i_unique.append(unique)
      self.iteration_i_counts.append(counts)

    if draw_method == "random":
      winners = sorted(zip(counts, unique), reverse=True)
      split_data_to_groups_by_count = []
      currentGroup = []
      last_count = winners[0][0]
      for countVote in winners:
        if last_count == countVote[0]:
          currentGroup.append(countVote[1])
        else:
          last_count = countVote[0]
          split_data_to_groups_by_count.append(currentGroup)
          currentGroup = [countVote[1]]
      if currentGroup != []:
        split_data_to_groups_by_count.append(currentGroup)

      if debug:
        print("Split_data : " , split_data_to_groups_by_count)

      num_of_winners_until_now = 0
      ret = []
      for gruopCount in split_data_to_groups_by_count:
        if num_of_winners_until_now == num_of_winners:
          return ret
        num_of_over_winners_in_group = len(gruopCount) + num_of_winners_until_now - num_of_winners
        if num_of_over_winners_in_group > 0:
          num_of_winners_to_add = num_of_winners - num_of_winners_until_now
          while (num_of_winners_to_add > 0):
            random_num = random.choice(gruopCount)
            if debug:
              print("Random Number : ", random_num)
            gruopCount.remove(random_num)
            ret.append(random_num)
            num_of_winners_to_add -= 1
          return ret
        else:
          num_of_winners_until_now += len(gruopCount)
          ret.extend(gruopCount)

    return [x for _, x in sorted(zip(counts, unique), reverse=True)][:num_of_winners]

  def train(self):
    self.voters_sampled_rank_by_partition = []
    self.Voter_partition_i = []
    Voter_partition = np.array(range(1,self.num_of_voters+1))
    np.random.shuffle(Voter_partition)
    if debug:
      print("num_of_voters : ", self.num_of_voters, " num_of_partitions :", self.num_of_partitions)
    num_of_voters_in_group = self.num_of_voters//self.num_of_partitions
    for partition_index in range(self.num_of_partitions):
      Voter_partition_i = Voter_partition[partition_index*num_of_voters_in_group:(partition_index+1)*num_of_voters_in_group]
      Voter_partition_i = sorted(Voter_partition_i, key=self.truthful_ranking.tolist().index) ##.sort sort by thruthful 
      self.Voter_partition_i.append(Voter_partition_i)
      if debug:
        print("Voter_partition[", partition_index , "] : ", Voter_partition_i)
      self.voters_sampled_rank_by_partition.append([[b for b in self.voter_sampled_ranks[x-1] if b not in Voter_partition_i] for x in Voter_partition_i]) ## At x-1 we have the rank for voter x

  def reSampled(self):
    self.voter_sampled_ranks = [create_n_rannking_from_truthful_ranking_and_sigma(truthful_ranking=np.delete(truthful_ranking, np.argwhere(truthful_ranking == voter)), sigma=sigma) for voter in range(1,num_of_voters+1)] ## Init ranks for each voter

  def get_rotate_array(self,original_list,num_of_rotate):
    copied_list = original_list[:]
    new_array = original_list[num_of_rotate:] + original_list[:num_of_rotate]
    return new_array

  def vote_same_votes(self):
    partition_winners = []
    partitions_score = {}
    votes_map = {}
    for i in range(self.num_of_reviews_per_candidate):
      for partition_index in range(self.num_of_partitions):
        #print(self.voters_sampled_rank_by_partition[partition_index])
        # print(self.Voter_partition_i[partition_index])
        if i == 0:
          partitions_score[partition_index] = {}
        voter_partition = self.get_rotate_array(self.Voter_partition_i[self.num_of_partitions-partition_index-1], i)
        for candidate, voter_rank, voter_name in zip(voter_partition,self.voters_sampled_rank_by_partition[partition_index],self.Voter_partition_i[partition_index]):
          vote = (1-(voter_rank.index(candidate)/len(voter_rank)))*100
          if i == 0:
            partitions_score[partition_index][candidate] = {}
            partitions_score[partition_index][candidate]['votes'] = [vote]
            partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/self.num_of_reviews_per_candidate
            partitions_score[partition_index][candidate]['rank_by'] = [voter_name]
          else:
            partitions_score[partition_index][candidate]['votes'].append(vote)
            partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/self.num_of_reviews_per_candidate
            partitions_score[partition_index][candidate]['rank_by'].append(voter_name)
    # print(self.voters_sampled_rank_by_partition[0])
    # print(partitions_score[0])
    # print(partitions_score[1])
    partition_0_winners = list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"])[0:self.num_of_winners_per_partition]))
    partition_1_winners = list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"])[0:self.num_of_winners_per_partition]))
    partition_winners.extend(partition_0_winners)
    partition_winners.extend(partition_1_winners)
    return partition_winners

  def get_not_vote_list(self,partitions_score, num_of_winners, num_of_losers, prob):
    not_rank_second_round = []
    partition_winners_first_round = []
    partition_losers_first_round = []


    sorted_partition_0 = list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"])))
    sorted_partition_1 = list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"])))
    
    partition_0_winners = sorted_partition_0[0:num_of_winners]
    partition_1_winners = sorted_partition_1[0:num_of_winners]
    need_to_pop = random.random()
    if need_to_pop < prob and num_of_winners > 0:
      partition_0_winners.pop()
      partition_1_winners.pop()


    partition_0_losers = sorted_partition_0[-1*num_of_losers:]
    partition_1_losers = sorted_partition_1[-1*num_of_losers:]
    partition_winners_first_round.extend(partition_0_winners)
    partition_winners_first_round.extend(partition_1_winners)

    partition_losers_first_round.extend(partition_0_losers)
    partition_losers_first_round.extend(partition_1_losers)

    not_rank_second_round.extend(partition_winners_first_round)
    not_rank_second_round.extend(partition_losers_first_round)

    return not_rank_second_round


  def vote_not_same_votes(self):
    partition_winners_first_round = []
    partition_losers_first_round = []
    partition_winners_second_round = []
    partition_winners = []
    not_rank_second_round = []
    partitions_score = {}
    num_of_first_votes = self.num_of_first_votes #math.floor(self.num_of_reviews_per_candidate*self.split_step)
    num_of_first_votes_winners = self.num_of_first_votes_winners #math.floor(self.num_of_winners*self.first_step_winners)
    num_of_first_votes_losers = self.num_of_first_votes_losers #math.floor(self.num_of_winners*self.first_step_losers)
    num_of_first_votes_winners_per_partition = self.num_of_first_votes_winners_per_partition #num_of_first_votes_winners//self.num_of_partitions
    num_of_first_votes_losers_per_partition = self.num_of_first_votes_losers_per_partition #num_of_first_votes_losers//self.num_of_partitions
    num_of_second_votes = self.num_of_second_votes #self.num_of_reviews_per_candidate - num_of_first_votes

    #print("Num of first votes winners: ",num_of_first_votes_losers_per_partition)
    #print("Num of first step rounds: ", num_of_first_votes)

    for i in range(num_of_first_votes):
      for partition_index in range(self.num_of_partitions):
        if i == 0:
          partitions_score[partition_index] = {}
        # print(self.Voter_partition_i[self.num_of_partitions-partition_index-1])
        voter_partition = self.get_rotate_array(self.Voter_partition_i[self.num_of_partitions-partition_index-1], i)
        for candidate, voter_rank, voter_name in zip(voter_partition,self.voters_sampled_rank_by_partition[partition_index],self.Voter_partition_i[partition_index]):
          vote = (1-(voter_rank.index(candidate)/len(voter_rank)))*100
          if i == 0:
            partitions_score[partition_index][candidate] = {}
            partitions_score[partition_index][candidate]['votes'] = [vote]
            partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/num_of_first_votes
            partitions_score[partition_index][candidate]['rank_by'] = [voter_name]
            partitions_score[partition_index][candidate]['num_of_votes'] = 1
          else:
            partitions_score[partition_index][candidate]['votes'].append(vote)
            partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/num_of_first_votes
            partitions_score[partition_index][candidate]['rank_by'].append(voter_name)
            partitions_score[partition_index][candidate]['num_of_votes'] += 1
    # print(partitions_score[0])
    # print(partitions_score[1])

    #print(list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"]))))
    #print(list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"]))))
    sorted_partition_0 = list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"])))
    sorted_partition_1 = list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"])))
    # print(sorted_partition_0)
    partition_0_winners = sorted_partition_0[0:num_of_first_votes_winners_per_partition]
    partition_1_winners = sorted_partition_1[0:num_of_first_votes_winners_per_partition]
    partition_0_losers = sorted_partition_0[-1*num_of_first_votes_losers_per_partition:]
    partition_1_losers = sorted_partition_1[-1*num_of_first_votes_losers_per_partition:]
    partition_winners_first_round.extend(partition_0_winners)
    partition_winners_first_round.extend(partition_1_winners)

    partition_losers_first_round.extend(partition_0_losers)
    partition_losers_first_round.extend(partition_1_losers)

    not_rank_second_round.extend(partition_winners_first_round)
    not_rank_second_round.extend(partition_losers_first_round)

    prob_for_not_rank = self.prob_for_not_rank
    
    # print("Winners first round: ", partition_winners_first_round)
    # print("Losers first round: ", partition_losers_first_round)
    # print("Not rank: ", not_rank_second_round)

    # print("From function not rank: ", self.get_not_vote_list(partitions_score,num_of_first_votes_winners_per_partition,num_of_first_votes_losers_per_partition))

    for i in range(num_of_first_votes,self.num_of_reviews_per_candidate+1):
      for partition_index in range(self.num_of_partitions):
        voter_partition = self.get_rotate_array(self.Voter_partition_i[self.num_of_partitions-partition_index-1], i)
        choosed_voters = []
        for candidate, voter_rank, voter_name in zip(voter_partition,self.voters_sampled_rank_by_partition[partition_index],self.Voter_partition_i[partition_index]):
          num_of_retries = 0
          if (candidate in not_rank_second_round) or (voter_name in partitions_score[partition_index][candidate]['rank_by']):
            candidate = random.choice(voter_partition)
            while (voter_name in partitions_score[partition_index][candidate]['rank_by']) or (candidate in choosed_voters) or (candidate in not_rank_second_round):
              num_of_retries+=1
              candidate = random.choice(voter_partition)
              if (num_of_retries == 10):
                break
              # print("Candidate: ", candidate)
              # print("Voter_name: ", voter_name, " Rank_by: ", partitions_score[partition_index][candidate]['rank_by'])
              # print("choosed_voters: ", choosed_voters)
              # print("not_rank_second_round: ", not_rank_second_round)
            if (num_of_retries == 10):
              continue
            choosed_voters.append(candidate)
          if (num_of_retries == 10):
            print("Bugg ", voter_name)
          vote = (1-(voter_rank.index(candidate)/len(voter_rank)))*100
          partitions_score[partition_index][candidate]['votes'].append(vote)
          partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/num_of_first_votes
          partitions_score[partition_index][candidate]['rank_by'].append(voter_name)
          partitions_score[partition_index][candidate]['num_of_votes'] += 1
      # print("not_rank_second_round ", i , " : ", not_rank_second_round)
      # print("\n", self.voters_sampled_rank_by_partition[0])
      # print(partitions_score[0],"\n")
      # print("\n",self.voters_sampled_rank_by_partition[1])
      # print(partitions_score[1],"\n")
      not_rank_second_round = self.get_not_vote_list(partitions_score,num_of_first_votes_winners_per_partition,num_of_first_votes_losers_per_partition,prob_for_not_rank)

    #print(partitions_score[0])
    #print(partitions_score[1])
    sorted_partition_0 = list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"])))
    sorted_partition_1 = list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"])))

    partition_0_winners = sorted_partition_0[0:self.num_of_winners_per_partition]
    partition_1_winners = sorted_partition_1[0:self.num_of_winners_per_partition]
    partition_winners.extend(partition_0_winners)
    partition_winners.extend(partition_1_winners)


    return partition_winners

  # def vote(self):
  #   if self.new_method:
  #     return self.vote_not_same_votes()
  #   else:
  #     return self.vote_same_votes()

  def best_vote(self):
    winners = []
    for partition_index in range(self.num_of_partitions):
      winners.extend(self.Voter_partition_i[partition_index][0:4])
    
    return winners

  def votes_all_candidates(self):
    partition_winners = []
    partitions_score = {}
    for partition_index in range(self.num_of_partitions):
      partitions_score[partition_index] = {}
      i=0
      for voter_rank, voter_name in zip(self.voters_sampled_rank_by_partition[partition_index],self.Voter_partition_i[partition_index]):
          for candidate in voter_rank:
            if i==0:
              partitions_score[partition_index][candidate] = {}
              partitions_score[partition_index][candidate]['votes'] = []
              partitions_score[partition_index][candidate]['mean'] = 0
              partitions_score[partition_index][candidate]['rank_by'] = []
              partitions_score[partition_index][candidate]['num_of_votes'] = 0

            vote = (1-(voter_rank.index(candidate)/len(voter_rank)))*100
            partitions_score[partition_index][candidate]['votes'].append(vote)
            partitions_score[partition_index][candidate]['mean'] = statistics.mean(partitions_score[partition_index][candidate]['votes']) #vote/num_of_first_votes
            partitions_score[partition_index][candidate]['rank_by'].append(voter_name)
            partitions_score[partition_index][candidate]['num_of_votes'] += 1
          i+=1
    #print("partition i: " ,self.voters_sampled_rank_by_partition[partition_index])
    sorted_partition_0 = list(map(lambda x: x[0],sorted(partitions_score[0].items(), key=lambda item: 100- item[1]["mean"])))
    sorted_partition_1 = list(map(lambda x: x[0],sorted(partitions_score[1].items(), key=lambda item: 100- item[1]["mean"])))


    #print("num of winners per partition: ", self.num_of_winners_per_partition)
    partition_0_winners = sorted_partition_0[0:self.num_of_winners_per_partition]
    partition_1_winners = sorted_partition_1[0:self.num_of_winners_per_partition]
    partition_winners.extend(partition_0_winners)
    partition_winners.extend(partition_1_winners)

    #print("All vote partition 0: ", self.voters_sampled_rank_by_partition[0])
    #print("All vote partition 1: ", self.voters_sampled_rank_by_partition[1])

    #print("All vote: ", partitions_score[0])
    #print("All vote: ", partitions_score[1])
    return partition_winners

  def vote(self, get_best_performe=False):
    return (self.best_vote(), self.votes_all_candidates(), self.vote_not_same_votes(), self.vote_same_votes())
    if self.new_method:
      return self.vote_not_same_votes()
    else:
      return self.vote_same_votes()

  def train_vote(self, get_best_performe=False):
    self.train()
    return self.vote(get_best_performe)

  def train_vote_x_times(self, num_of_iterations):
    iterations_votes = [self.train_vote() for _ in range(num_of_iterations)]  ##runs num_of_iterations different partition and store their votes
    return self.get_partition_winners(iterations_votes)

  def train_vote_x_times_score(self, num_of_iterations,best_method=[],metrics=["avg","dis", "avgDis"],resmapled=True):
    if resmapled:
      self.reSampled()  ##resampled voters ranks 
    if not len(best_method):
      best_method = self.best_method
    ret = {}
    for metric in metrics: ##Init ret
      ret[metric] = 0

    iterations_votes = [self.train_vote() for _ in range(num_of_iterations)]  ##runs num_of_iterations different partition and store their vote
    self.itreration_train_vote_x_times.append(self.get_partition_winners(iterations_votes,keep_log=True))

    for i in range(num_of_iterations):
      acc_dic = self.accuray(iterations_votes[i],best_method=best_method,metrics=metrics)
      for metric in metrics:
        ret[metric] += acc_dic[metric]

    for metric in metrics:
        ret[metric] = ret[metric]/num_of_iterations

    return ret

  def train_vote_x_times_score_resmapled(self, num_of_iterations,resmapled_every=5,best_method=[],metrics=["avg","dis", "avgDis"]):
    if not len(best_method):
      best_method = self.best_method

    ret = {}
    for metric in metrics: ##Init ret
      ret[metric] = 0

    for _ in range(resmapled_every):
      x_time_score = self.train_vote_x_times_score(num_of_iterations=num_of_iterations,best_method=best_method,metrics=metrics,resmapled=True)
      print("x_time_score: ", x_time_score)
      for metric in metrics:
        ret[metric] += x_time_score[metric]

    for metric in metrics:
        ret[metric] = ret[metric]/resmapled_every

    return ret
    
    #iterations_votes_scores = [self.train_vote_x_times_score(num_of_iterations=num_of_iterations,best_method=best_method,metrics=metrics,resmapled=True) for _ in range(resmapled_every)]  ##runs num_of_iterations different partition and store their votes
    #return sum(iterations_votes_scores)/num_of_iterations

  def calc_distance_between_votes(self, first_vote,second_vote):
    return self.truthful_ranking.tolist().index(first_vote) - self.truthful_ranking.tolist().index(second_vote)

  def accuracy(self, votes, best_method=[],metrics=["avg","dis", "avgDis"]):
    if not len(best_method):
      best_method = self.best_method
    ret = {}
    wrong_votes = [x for x in votes if x not in best_method]
    missing_best = [x for x in best_method if x not in votes]
    num_of_winners = len(best_method)
    #print("self.num_of_winners: ", self.num_of_winners)
    missing_votes_avg = len(wrong_votes)/num_of_winners
    if "avg" in metrics:
      ret["avg"] = missing_votes_avg

    if "dis" in metrics or "avgDis" in metrics:
      distance_on_error = 0
      for wrong_vote, miss_vote in zip(wrong_votes, missing_best):
        distance_between_vote = self.calc_distance_between_votes(wrong_vote, miss_vote)
        distance_on_error+= distance_between_vote
      if "dis" in metrics:
        ret["dis"] = distance_on_error
      if "avgDis" in metrics:
        ret["avgDis"] = missing_votes_avg*distance_on_error
    
    return ret

  def print_test_summary(self):
    print("\nSummary ")
    print("\nphi: ", self.sigma)
    print("\nnum of voters: ", self.num_of_voters)
    print("\nnum of winners: ", self.num_of_winners)
    print("\nnum_of_reviews_per_candidate: ", self.num_of_reviews_per_candidate)
    print("\nnum of first votes: ", self.num_of_first_votes)
    print("\nnum_of_second_votes: ", self.num_of_second_votes)
    print("\nnum_of_first_votes_winners: ",self.num_of_first_votes_winners)
    print("\nnum_of_first_votes_losers: ", self.num_of_first_votes_losers)
    print("\nprob_for_not_rank: ", self.prob_for_not_rank)
    print("\n\n\n")
