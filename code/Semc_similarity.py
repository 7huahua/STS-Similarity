# import libraries
import pandas as pd
import numpy as np
from prefixspan import PrefixSpan
# import kmeans from sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer



class SemcSimilarity(object):
    # init
    def __init__(self,stayregions=None):
        self.stayregions = stayregions
        self.stayregions = self.set_day_week_weekday(self.stayregions)

    def set_day_week_weekday(self,stayregions,time = 'arr_t'):
        stayregions['week'] = stayregions[time].dt.week
        stayregions['weekday'] = stayregions[time].dt.weekday
        stayregions['day'] = stayregions[time].dt.day
        return stayregions

    # get poi tfidf
    def get_poi_tfidf(self,stayregions,poi_label = 'poi'):
        # for each stay region, treat it as a phase, and the pois within it as words
        # get the tfidf of each phase
        corpus = stayregions[poi_label].apply(lambda x: ' '.join(x)).values
        vectorizer = TfidfVectorizer(stop_words = None,token_pattern='(?u)\\b\\w+\\b')
        sparse_matrix = vectorizer.fit_transform(corpus)
        print(type(sparse_matrix))
        print(sparse_matrix.shape)

        return sparse_matrix

    # cluster stay regions by poi tfidf
    def cluster_stayregions(self,stayregions,label):
        # get poi tfidf
        poi_tfidf = self.get_poi_tfidf(stayregions)
        # cluster stay regions by poi tfidf value using kmeans
        kmeans = KMeans(n_clusters=3, random_state=0).fit(poi_tfidf)
        stayregions[label] = kmeans.labels_
        return stayregions

    # get stay region sequence by week
    def by_week(self,stayregions,label):
        by_week = {}
        for user_id,group in stayregions.groupby('user_id'):
            week_squences = []
            for week,week_group in group.groupby('week'):
                week_squences.append(week_group[label].tolist())
            by_week[user_id] = week_squences
        return by_week
    
    # get stay region sequence by weekday
    def by_weekday(self,stayregions,label):
        by_weekday = {}
        for user_id,group in stayregions.groupby('user_id'):
            weekday_squences = []
            for weekday,weekday_group in group.groupby('weekday'):
                weekday_squences.append(weekday_group[label].tolist())
            by_weekday[user_id] = weekday_squences
        return by_weekday
    
    # get stay region sequence by week with daily subset
    def by_week_with_subset(self,stayregions,label):
        by_week_with_subset = {}
        for user_id,group in stayregions.groupby('user_id'):
            week_squences = []
            for week,week_group in group.groupby('week'):
                day_sequences = []
                for day,day_group in week_group.groupby('day'):
                    day_sequences.append(day_group[label].tolist())

                week_squences.append(day_sequences)
            by_week_with_subset[user_id] = week_squences
        return by_week_with_subset
    
    # get stay region sequence by weekday with daily subset
    def by_weekday_with_subset(self,stayregions,label):
        by_weekday_with_subset = {}
        for user_id,group in stayregions.groupby('user_id'):
            weekday_squences = []
            for weekday,weekday_group in group.groupby('weekday'):
                day_sequences = []
                for day,day_group in weekday_group.groupby('day'):
                    day_sequences.append(day_group[label].tolist())

                weekday_squences.append(day_sequences)
            by_weekday_with_subset[user_id] = weekday_squences
        return by_weekday_with_subset

    def get_stayregion_sequence(self,stayregions,label = 'cluster_id'):
        # for ever user, for every week, get stay region sequence with label
        # stay regions within one day is a turple

        # get different segmentation sequences
        self.by_week = self.by_week(stayregions,label)
        self.by_weekday = self.by_weekday(stayregions,label)
        self.by_week_with_subset = self.by_week_with_subset(stayregions,label)
        self.by_weekday_with_subset = self.by_weekday_with_subset(stayregions,label)

    # extract prefixspan
    def extract_prefixspan(self,stayregions,label):
        pass

    # extract max frequent sequences
    def extract_max_frequent_sequence(self,stayregions):
        pass


if __name__ == '__main__':

    # Input dataset: a list of sequences, where each sequence is a list of itemsets
    # Each itemset is a list of items
    dataset = [
        [(1, 2), (3)],
        [(1), (3, 2), (1, 2)],
        [(1, 2), (5)],
        [(6)]
    ]
    
    # create a dataset only containing item not itemset
    dataset = [[1,2,3],[1,3,2,1,2],[1,2,5],[6]]

    # Instantiate the PrefixSpan class with the dataset
    ps = PrefixSpan(dataset)

    # Set the minimum support threshold
    min_support = 2

    # Mine frequent sequential patterns
    frequent_patterns = ps.frequent(min_support)

    # Print the frequent patterns along with their support count
    for support, pattern in frequent_patterns:
        print(f"Pattern: {pattern}, Support: {support}")
        

    # Output:Pattern: 3, Support: [(1, 2)]
