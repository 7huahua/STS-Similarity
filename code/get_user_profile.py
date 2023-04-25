# import libraries
import pandas as pd
import numpy as np
from prefixspan import PrefixSpan
# import kmeans from sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pickle
from get_semc_sequence import SemcSequence

class UserProfile(object):
    # init
    def __init__(self,stayregions=None):
        # get stay region sequence
        semc_sequences = SemcSequence(stayregions)
        self.stayregions = semc_sequences.stayregions
        self.sequence_by_week = semc_sequences.sequence_by_week
        self.sequence_by_weekday = semc_sequences.sequence_by_weekday
        # mfp means maximal frequent pattern
        self.week_mfp_dict= self.extract_prefixspan(self.sequence_by_week)
        self.weekday_mfp_dict = self.extract_prefixspan(self.sequence_by_weekday)

    # extract prefixspan
    def extract_prefixspan(self,sequences):
        # sequences is a dict, key is user_id, value is a list of sequences
        # use a dict to store the frequent patterns
        max_frequent_pattern_dict = {}
        # for every user, get prefixspan
        for user_id,seq in sequences.items():
            ps = PrefixSpan(seq)
            # Set the minimum support threshold
            min_support = 1

            # Mine frequent sequential patterns
            frequent_patterns = ps.frequent(min_support)
            
            # # 如果没有找到maximal frequent pattern，就把min_support减小一点
            # if len(frequent_patterns) == 0:
            #     min_support = 1
            #     frequent_patterns = ps.frequent(min_support)
            #     maximal_frequent_patterns = self.find_max_frequent_patterns(frequent_patterns)

            # # 如果maximal frequent pattern大于1000，就把min_support增大一点
            # el
            # if len(frequent_patterns) > 10000:
            #     min_support = 3
            #     frequent_patterns = ps.frequent(min_support)
            #     maximal_frequent_patterns = self.find_max_frequent_patterns(frequent_patterns)

            # else:
            #     
            maximal_frequent_patterns = self.find_max_frequent_patterns(frequent_patterns)
            
            print('user {} has {} frequent patterns, and {} maximal frequent patterns'.format(user_id,len(frequent_patterns),len(maximal_frequent_patterns)))
            # print(maximal_frequent_patterns)
            max_frequent_pattern_dict[user_id] = maximal_frequent_patterns

        return max_frequent_pattern_dict


    # extract max frequent sequences
    def find_max_frequent_patterns(self,frequent_patterns):
        max_frequent_patterns = []
        
        # Sort the patterns based on their support in descending order
        sorted_patterns = sorted(frequent_patterns, key=lambda x: (-x[0], x[1]))

        for pattern in sorted_patterns:
            is_max = True
            support, itemset = pattern
            
            for max_pattern in max_frequent_patterns:
                _, max_itemset = max_pattern
                
                # Check if the current pattern is a subset of any max_pattern
                if set(itemset).issubset(set(max_itemset)):
                    is_max = False
                    break
            
            if is_max:
                max_frequent_patterns.append(pattern)

        # Perform an additional check to ensure max_frequent_patterns are indeed maximal
        final_max_frequent_patterns = []
        for pattern in max_frequent_patterns:
            support, items = pattern
            is_maximal = True

            for other_pattern in max_frequent_patterns:
                other_support, other_items = other_pattern
                if items != other_items and set(items).issubset(set(other_items)):
                    is_maximal = False
                    break

            if is_maximal:
                final_max_frequent_patterns.append(pattern)

        return final_max_frequent_patterns
    
    # extract max frequent sequences and max support patterns
    # 这里有必要说一下，这是chatgpt提供的一种方法，用来快速获得最大频繁模式
    # 然而我认为，也许保留部分最频繁pattern也是有必要的。这些可能包含了home 以及 workplace
    def find_max_frequent_support_patterns(self,frequent_patterns):
        max_frequent_patterns = []
        
        # Sort the patterns based on their support in descending order
        sorted_patterns = sorted(frequent_patterns, key=lambda x: (-x[0], x[1]))

        for pattern in sorted_patterns:
            is_max = True
            support, itemset = pattern
            
            for max_pattern in max_frequent_patterns:
                _, max_itemset = max_pattern
                
                # Check if the current pattern is a subset of any max_pattern
                if set(itemset).issubset(set(max_itemset)):
                    is_max = False
                    break
            
            if is_max:
                max_frequent_patterns.append(pattern)

        return max_frequent_patterns
    
# test
if __name__ == '__main__':
    # first get stay regions
    stayregions = pd.read_csv('data/user_profile/resampled_data_dropped_bymean_bylocal.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]
    # drop user 29 and user 229
    stayregions = stayregions[stayregions['user']!=29]
    stayregions = stayregions[stayregions['user']!=229]

    print(stayregions.shape)

    # get stay region sequence
    user_profile = UserProfile(stayregions)
    # print result
    print(user_profile.stayregions.head())

    # extract prefixspan
    week_max_frequent_pattern_dict = user_profile.week_mfp_dict
    weekday_max_frequent_pattern_dict = user_profile.weekday_mfp_dict

    # # save stay regions to pandas
    # with open("./data/user_profile/local_resampled_stayregions_bymean.pkl", "wb") as f:
    #     pickle.dump(user_profile.stayregions, f)
    
    with open("./data/user_profile/dropped_mean_local_week_sup1.pkl", "wb") as f:
        pickle.dump(week_max_frequent_pattern_dict, f)
    
    with open("./data/user_profile/dropped_mean_local_weekday_sup1.pkl", "wb") as f:
        pickle.dump(weekday_max_frequent_pattern_dict, f)