import pandas as pd
import numpy as np
from Semc_similarity import SemcSimilarity
from ST_similarity import STSimilarity
from sklearn.metrics import average_precision_score, ndcg_score
from scipy.stats import rankdata
from matplotlib import pyplot as plt

class TotalSimilarity():

    def compute_total_similarity(self,st_sim=None,semc_sim=None,w1=0.5,w2=0.5):
        # total similarity is the average of semc similarity and st similarity
        # st_sim is a nested dict, key is [user1][user2], value is similarity
        # semc_sim is a nested dict, key is [user1][user2], value is similarity
        # w1 is the weight of st similarity
        # w2 is the weight of semc similarity
        # return a dict, key is [user1][user2], value is similarity
        total_sim = {}

        # if st_sim is not given, only use semc similarity
        if st_sim is None:
            self.total_sim_dict = semc_sim
            return semc_sim
        
        # if semc_sim is not given, only use st similarity
        if semc_sim is None:
            self.total_sim_dict = st_sim
            return st_sim
        

        # first, select the different users in st_sim and semc_sim
        st_sim_users = set([user1 for user1 in st_sim.keys()])
        semc_sim_users = set([user1 for user1 in semc_sim.keys()])
        diff_users = st_sim_users.symmetric_difference(semc_sim_users)
        print(diff_users)

        # ignore the different users

        # then, compute the total similarity
        # total_sim[user1][user2] = w1*st_sim[user1][user2] + w2*semc_sim[user1][user2]
        for user1 in st_sim.keys():
            if user1 in diff_users:
                continue
            if user1 not in total_sim:
                total_sim[user1] = {}
            for user2 in st_sim[user1].keys():
                if user2 in diff_users:
                    continue
                if user1 != user2:
                    total_sim[user1][user2] = w1*st_sim[user1][user2] + w2*semc_sim[user1][user2]
                else:
                    total_sim[user1][user2] = 0.0

        self.total_sim_dict = total_sim

    def get_similar_users(self,total_sim,k):
        pass
        # get similar users from total similarity
        # total_sim is a nested dict, key is [user1][user2], value is similarity
        # return a dict, key is user, value is a list of similar users
        
        # total_sim is a nested dict, key is [user1][user2], value is similarity
        # self.max_sim_neighbors = {user: max(sim_dict, key=sim_dict.get) for user, sim_dict in total_sim.items()}
        self.k_sim_neighbors = {user: sorted(sim_dict, key=sim_dict.get, reverse=True)[:k] for user, sim_dict in total_sim.items()}
        return self.k_sim_neighbors
    
    def get_ground_truth(self,stayregions):
        # get ground truth 
        # stayregions is a nested dict, key is user, value is a dict
        # return a nested dict, key is user, value is a dict
        ground_truth = {}
        user_list = stayregions.user.unique()
        for user1 in user_list:
            ground_truth[user1] = {}
            for user2 in user_list:
                if user1 == user2+200 and user1 >= 200:
                    ground_truth[user1][user2] = 1
                elif user1 == user2-200 and user1 < 200:
                    ground_truth[user1][user2] = 1
                else:
                    ground_truth[user1][user2] = 0
        self.ground_truth = ground_truth
    
    def dict_to_array(self,nested_gt,nested_pred):
        user_list = list(nested_pred.keys())
        n_users = len(user_list)
        n_items = len(next(iter(nested_pred.values())))

        # create y_true and y_pred matrix
        y_true = np.zeros((n_users, n_items))
        y_pred = np.zeros((n_users, n_items))

        # fullfill y_true and y_pred matrix
        for i, user in enumerate(user_list):
            gt_dict = nested_gt[user]
            pred_dict = nested_pred[user]
            
            # sort the dict by value
            sorted_items = sorted(pred_dict, key=pred_dict.get, reverse=True)
            
            for j, item in enumerate(sorted_items):
                y_true[i, j] = gt_dict[item]
                y_pred[i, j] = pred_dict[item]

        print("y_true:", y_true)
        print("y_pred:", y_pred)

        return y_true, y_pred
    
    def evaluate(self, y_true,y_pred,k):

        # select top k items
        y_true_k = y_true[:, :k]
        y_pred_k = y_pred[:, :k]

        MAP_k = self.compute_map(y_true_k, y_pred_k)
        print("MAP@k:", MAP_k)

        NDCG_k = self.compute_ndcg(y_true_k, y_pred_k)
        print("NDCG@k:", NDCG_k)
        
        return MAP_k, NDCG_k


    def plot_multiple_lines(self, x_values, y_values_list, labels, colors, markers, linestyles,filename=None):
        plt.figure(figsize=(10, 6))
        
        for i, y_values in enumerate(y_values_list):
            plt.plot(x_values, y_values, label=labels[i], color=colors[i], marker=markers[i], linestyle=linestyles[i])

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # set xticks
        plt.xticks(np.arange(1, 21, 1))
        plt.title('/'.join(label for label in labels))
        plt.legend(loc='best')
        plt.show()
        if filename:
            plt.savefig(filename)
        


    def compute_map(self,y_true, y_pred):
        # y_pred is a sorted list of predicted items
        # y_true is a list of ground truth items
        # k is the number of items to be recommended
        ap = 0
        for i in range(y_true.shape[0]):
            # print("y_true:", y_true[i])
            # print("y_pred:", y_pred[i])
            # if the ground truth item is in the top k predicted items
            # compute the rank of the predicted item
            rank = rankdata(y_pred[i], method='dense')
            max_rank = np.max(rank)
            reversed_rank = max_rank - rank + 1

            # if the 1 in y_true's rank is smaller that k
            # add the precision to the average precision
            
            # get the index of 1 in y_true
            y_true_index = np.where(y_true[i] == 1)[0]
            if y_true_index!=None:
                ap += 1/(reversed_rank[y_true_index])
                # print("rank:", reversed_rank[y_true_index])
                # print("ap:", ap)
                

        return ap/y_true.shape[0]

    def compute_ndcg(self,y_true, y_pred):
        # y_pred is a sorted list of predicted items
        # y_true is a list of ground truth items
        # k is the number of items to be recommended
        ap = 0
        for i in range(y_true.shape[0]):
            # print("y_true:", y_true[i])
            # print("y_pred:", y_pred[i])
            # if the ground truth item is in the top k predicted items
            # compute the rank of the predicted item
            rank = rankdata(y_pred[i], method='dense')
            max_rank = np.max(rank)
            reversed_rank = max_rank - rank + 1

            y_true_index = np.where(y_true[i] == 1)[0]
            if y_true_index!=None:
                ap += 1/np.log2(reversed_rank[y_true_index]+1)

        return ap/y_true.shape[0]
        