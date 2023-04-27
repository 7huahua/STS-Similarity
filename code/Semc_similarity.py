# import libraries
import pandas as pd
import numpy as np
import pickle
from get_user_profile import UserProfile

class SemcSimilarity(object):
    # init
    def __init__(self,stayregions=None,profile_path=None):
        # if profile_path is not None, load user_profile from profile_path
        if profile_path:
            with open(profile_path,'rb') as f:
                user_profile = pickle.load(f)
                self.user_profile = user_profile

        # if profile_path is None, compute user_profile from stayregions
        else:
            user_profile = UserProfile(stayregions)
            self.stayregions = user_profile.stayregions
            self.user_profile = user_profile.week_mfp_dict
        # self.user_weekday_profiles = user_profile.weekday_mfp_dict
    
    # finally, compute semc similarity
    def compute_semc_similarity(self,profiles):
        # for every 2 user, compute semc similarity based on their patterns
        # profiles is a dict, key is user_id, value is a list of patterns
        # use a nested dict to store the similarity
        similarity_dict = {}
        for user1,profile1 in profiles.items():
            if user1 not in similarity_dict:
                similarity_dict[user1] = {}
            for user2,profile2 in profiles.items():
                if user1 != user2:
                    # compute similarity
                    similarity = self.compute_prof_similarity(profile1,profile2)
                    # store similarity
                    # similarity_dict[(user1,user2)] = similarity
                    similarity_dict[user1][user2] = similarity
                    print('user {} to user {}, similarity is {}'.format(user1,user2, similarity))
                else:
                    # similarity_dict[(user1,user2)] = 1.0
                    similarity_dict[user1][user2] = 0.0

        return similarity_dict
    
    def compute_prof_similarity(self,profile1,profile2):
        # compute similarity of user 1 to user 2
        # profile is a list of patterns

        # if profile1 or profile2 is empty, return 0.0
        if len(profile1) == 0 or len(profile2) == 0:
            return 0.0

        # compute similarity
        similarity = 0.0
        
        denom = 0.0
        numer = 0.0
        for sup1, pattern1 in profile1:
            # denominator = sum of (sup1*length)
            denom += sup1 * len(pattern1)
            lcs = []
            
            for sup2, pattern2 in profile2:
                # print(sup2,pattern2)
                # numerinator is sum of profile1 and profile2's longest common subsequence 's（sup*length)
                new_lcs = self.longest_common_subsequence(pattern1,pattern2)
                if len(new_lcs) == min(len(pattern1),len(pattern2)):
                    # if new_lcs is the same length of one of the pattern, break
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



    

