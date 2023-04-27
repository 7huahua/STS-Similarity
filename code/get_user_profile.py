# import libraries
import pandas as pd
from prefixspan import PrefixSpan
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
            min_support = 2

            # Mine frequent sequential patterns
            frequent_patterns = ps.frequent(min_support)

            # if the number of frequent patterns is too large
            # please increase the min_support or resample users
              
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
    # this function is not used, as some large support patterns are also included
    # however, those patterns could include home and workplace
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
    stayregions = pd.read_csv('data/user_profile/stayregions_sample.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]
    print(stayregions.shape)

    # get stay region sequence
    user_profile = UserProfile(stayregions)

    # extract prefixspan
    pattern_dict = user_profile.week_mfp_dict
    
    # save the results
    with open("./data/user_profile/profile_sample.pkl", "wb") as f:
        pickle.dump(pattern_dict, f)