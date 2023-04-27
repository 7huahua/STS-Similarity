import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import silhouette_score

class SemcSequence(object):
    # init
    def __init__(self,stayregions=None):
        self.stayregions = self.set_day_week_weekday(stayregions)
        # k of k-means is set to 35 the same as the other paper, you can change it to other values
        self.stayregions = self.cluster_stayregions(self.stayregions,'category1','cluster_id',k=35)
        # get stay region sequence
        self.semc_sequences = self.get_stayregion_sequence(self.stayregions)


    def set_day_week_weekday(self,stayregions,time = 'arr_t'):
        # week is the week from the 2007-04-09(the monday of this week), as the earliest time is 2007-04-12
        stayregions['week'] = (stayregions[time] - pd.to_datetime('2007-04-09')).dt.days // 7
        stayregions['weekday'] = stayregions[time].dt.weekday
        stayregions['day'] = stayregions[time].dt.day
        return stayregions

    # get poi tfidf
    def get_poi_tfidf(self,stayregions,poi_label):
        # for each stay region, treat it as a phase, and the pois within it as words
        corpus = [' '.join(poi.split(';')) for poi in stayregions[poi_label]]
        # print(corpus)
        # filter some stop words
        custom_stop_words = ['life','未知']
        vectorizer = TfidfVectorizer(stop_words = custom_stop_words,token_pattern='(?u)\\b\\w+\\b')
        sparse_matrix = vectorizer.fit_transform(corpus)
        # print all the words
        # word = vectorizer.get_feature_names_out()
        # print(word)
        # print(sparse_matrix.shape)

        return sparse_matrix

    # cluster stay regions by poi tfidf
    def cluster_stayregions(self,stayregions,poi_label,cluster_label, k):
        # get poi tfidf
        poi_tfidf = self.get_poi_tfidf(stayregions,poi_label)
        
        # cluster stay regions by poi tfidf value using kmeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(poi_tfidf)
        stayregions[cluster_label] = kmeans.labels_

        # print cluster label and its count
        # print(stayregions[cluster_label].value_counts()) # you can check the distribution of the cluster result here 

        # visualize cluster result, using the matrix of tfidf
        # self.visualize_cluster_result_2D(poi_tfidf.toarray(),stayregions[cluster_label].tolist())
        # self.visualize_cluster_result_3D(poi_tfidf.toarray(),stayregions[cluster_label].tolist())
        
        return stayregions
    
    def visualize_cluster_result_2D(self,vectors,labels):

        tsne_model = TSNE(n_components=2, perplexity=40, learning_rate=100, n_iter=500)

        tsne_vectors = tsne_model.fit_transform(vectors)

        unique_labels = list(set(labels))

        colors = sns.color_palette("husl", len(unique_labels))
        for i, label in enumerate(unique_labels):
            x = tsne_vectors[np.where(np.array(labels) == label), 0]
            y = tsne_vectors[np.where(np.array(labels) == label), 1]
            plt.scatter(x, y, label=str(label),color=colors[i],s=1)
        
        plt.legend()
        plt.show()

    def visualize_cluster_result_3D(self,vectors,labels):

        tsne_model = TSNE(n_components=3, perplexity=40, learning_rate=100, n_iter=500)
        tsne_vectors = tsne_model.fit_transform(vectors)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = list(set(labels))
        colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(unique_labels))]

        for i, label in enumerate(unique_labels):
            x = tsne_vectors[np.where(np.array(labels) == label), 0]
            y = tsne_vectors[np.where(np.array(labels) == label), 1]
            z = tsne_vectors[np.where(np.array(labels) == label), 2]
            ax.scatter(x, y, z, label=str(label), color=colors[i], s=1)

        ax.legend()
        plt.show()



    # get stay region sequence by week
    def by_week(self,stayregions,label):
        by_week = {}
        # sort stayregions by user and arrival time
        stayregions = stayregions.sort_values(by=['user','arr_t'])
        for user_id,group in stayregions.groupby('user'):
            week_squences = []
            for week,week_group in group.groupby('week'):
                week_squences.append(week_group[label].tolist())
            by_week[user_id] = week_squences
        return by_week
    
    # get stay region sequence by weekday
    def by_weekday(self,stayregions,label):
        by_weekday = {}
        stayregions = stayregions.sort_values(by=['user','arr_t'])
        for user_id,group in stayregions.groupby('user'):
            weekday_squences = []
            for weekday,weekday_group in group.groupby('weekday'):
                weekday_squences.append(weekday_group[label].tolist())
            by_weekday[user_id] = weekday_squences
        return by_weekday
    
    # get stay region sequence by week with daily subset
    def by_week_with_subset(self,stayregions,label):
        by_week_with_subset = {}
        stayregions = stayregions.sort_values(by=['user','arr_t'])
        for user_id,group in stayregions.groupby('user'):
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
        stayregions = stayregions.sort_values(by=['user','arr_t'])
        for user_id,group in stayregions.groupby('user'):
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
        self.sequence_by_week = self.by_week(stayregions,label)
        self.sequence_by_weekday = self.by_weekday(stayregions,label)
        self.sequence_by_week_with_subset = self.by_week_with_subset(stayregions,label)
        self.sequence_by_weekday_with_subset = self.by_weekday_with_subset(stayregions,label)


if __name__ == "__main__":
    # first get stay regions
    stayregions = pd.read_csv('data/user_profile/stayregions_sample.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]

    # get stay region sequence
    semc_sequence = SemcSequence(stayregions)
    semc_sequence.get_stayregion_sequence(stayregions)
    print(semc_sequence)
