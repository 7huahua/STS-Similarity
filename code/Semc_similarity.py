# import libraries
import pandas as pd
import numpy as np
import pickle
from get_user_profile import UserProfile

class SemcSimilarity(object):
    # init
    def __init__(self,stayregions=None,profile_path=None):
        # 如果传入了profile_path，就直接读取
        if profile_path:
            with open(profile_path,'rb') as f:
                user_profile = pickle.load(f)
                self.user_profile = user_profile

        # 如果没有传入profile_path，就重新计算用户profile
        user_profile = UserProfile(stayregions)
        self.stayregions = user_profile.stayregions
        self.user_profile = user_profile.week_mfp_dict
        # self.user_weekday_profiles = user_profile.weekday_mfp_dict
    
    # finally, compute semc similarity
    def compute_semc_similarity(self,profiles):
        # for every 2 user, compute semc similarity based on their patterns
        # profiles is a dict, key is user_id, value is a list of patterns
        # use a dict to store the similarity
        similarity_dict = {}
        for user1,profile1 in profiles.items():
            for user2,profile2 in profiles.items():
                if user1 != user2:
                    # compute similarity
                    similarity = self.compute_prof_similarity(profile1,profile2)
                    # store similarity
                    similarity_dict[(user1,user2)] = similarity
                    print('user {} to user {}, similarity is {}'.format(user1,user2, similarity))
                else:
                    similarity_dict[(user1,user2)] = 1.0

        return similarity_dict
    
    def compute_prof_similarity(self,profile1,profile2):
        # compute similarity of user 1 to user 2
        # profile is a list of patterns
        # compute similarity
        similarity = 0.0
        
        denom = 0.0
        numer = 0.0
        for sup1, pattern1 in profile1:
            # 分母为profile1的总（支持度*长度）之和
            # print(sup1,pattern1)
            denom += sup1 * len(pattern1)
            lcs = []
            
            for sup2, pattern2 in profile2:
                # print(sup2,pattern2)
                # 分子为profile1和profile2中longest common subsequence的（支持度*长度）之和
                new_lcs = self.longest_common_subsequence(pattern1,pattern2)
                if len(new_lcs) == min(len(pattern1),len(pattern2)):
                    # 如果lcs的长度等于较短的pattern的长度，那么就不用继续比较了
                    lcs = new_lcs
                    break
                if len(new_lcs) > len(lcs):
                    lcs = new_lcs
                
            numer += sup1 * len(lcs)
            # print(lcs)
            # print('denom is {}, numer is {}'.format(denom,numer))

        similarity = numer / denom

        return similarity
    
    def longest_common_subsequence(self,pattern1,pattern2):
        # compute longest common subsequence
        # pattern is a list of items
        # use dynamic programming
        # initialize dp
        dp = [[0 for _ in range(len(pattern2)+1)] for _ in range(len(pattern1)+1)]
        # compute dp
        for i in range(1,len(pattern1)+1):
            for j in range(1,len(pattern2)+1):
                if pattern1[i-1] == pattern2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        
        # get longest common subsequence
        lcs = []
        i = len(pattern1)
        j = len(pattern2)
        while i > 0 and j > 0:
            if pattern1[i-1] == pattern2[j-1]:
                lcs.append(pattern1[i-1])
                i -= 1
                j -= 1
            else:
                if dp[i-1][j] > dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
        
        return lcs[::-1]

if __name__ == '__main__':
    # # if need re-extract user profile
    # stayregions = pd.read_csv('data/combined_poi.csv',parse_dates=['arr_t','lea_t'])
    # # only remain rows with value of category1 or category2
    # stayregions = stayregions[stayregions['category1'].notnull()]
    # print(stayregions.shape)

    # # get stay region sequence
    # semc_similarity = SemcSimilarity(stayregions)
    # # print result
    # print(semc_similarity.stayregions.head())
    
    # or load user profile from file
    semc_similarity = SemcSimilarity(profile_path='data/user_profile/week_mfp_dict.pkl')
    # print result
    loaded_max_frequent_patterns_dict = semc_similarity.user_profile

    user_list = []
    pattern_num_list = []
    for user_id, patterns in loaded_max_frequent_patterns_dict.items():
        print('user {} has {} max frequent patterns'.format(user_id,len(patterns)))
        # print(patterns)
        # print()
        user_list.append(user_id)
        pattern_num_list.append(len(patterns))

    # check if user_id + or - 200 is in the list
    for user_id in user_list:
        if user_id < 200 and user_id + 200 in user_list:
            continue
        elif user_id >= 200 and user_id - 200 in user_list:
            continue
        else:
            print('user {} does not have round truth'.format(user_id))

    # check users with no frequent patterns
    for user_id in user_list:
        if len(loaded_max_frequent_patterns_dict[user_id]) == 0:
            print('user {} has no frequent patterns'.format(user_id))

    # check mean number of frequent patterns
    print('mean number of frequent patterns is {}'.format(np.mean(pattern_num_list)))

    # check max number of frequent patterns
    print('max number of frequent patterns is {}'.format(np.max(pattern_num_list)))

    # check median number of frequent patterns
    print('median of frequent patterns is {}'.format(np.median(pattern_num_list)))

    # if drop users with no frequent patterns, print the mean number of frequent patterns
    pattern_num_list = [x for x in pattern_num_list if x != 0]
    print('mean number of frequent patterns is {}'.format(np.mean(pattern_num_list)))
    print('median of frequent patterns is {}'.format(np.median(pattern_num_list)))

    # print the user and their frequent patterns whose number of frequent patterns is 1
    for user_id, patterns in loaded_max_frequent_patterns_dict.items():
        if len(patterns) == 1:
            print('user {} has {} frequent patterns'.format(user_id,len(patterns)))
            print(patterns)
            # print()

    # semc_similarity = SemcSimilarity()
    # compute similarity
    similarity_dict = semc_similarity.compute_semc_similarity(loaded_max_frequent_patterns_dict)
    print(similarity_dict)


    

