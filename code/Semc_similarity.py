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
import seaborn as sns
import pickle
from mpl_toolkits.mplot3d import Axes3D





class SemcSimilarity(object):
    # init
    def __init__(self,stayregions=None):
        self.stayregions = stayregions
        self.stayregions = self.set_day_week_weekday(self.stayregions)

    def set_day_week_weekday(self,stayregions,time = 'arr_t'):
        # week is the week from the 2007-04-09(the monday of this week), as the earliest time is 2007-04-12
        stayregions['week'] = (stayregions[time] - pd.to_datetime('2007-04-09')).dt.days // 7
        stayregions['weekday'] = stayregions[time].dt.weekday
        stayregions['day'] = stayregions[time].dt.day
        return stayregions

    # get poi tfidf

    def get_poi_tfidf(self,stayregions,poi_label):
        # for each stay region, treat it as a phase, and the pois within it as words
        # get the tfidf of each phase
        # 对于每一行的poi str，在使用分号分割后，用空格连接起来，形成一个字符串
        # 首先，对于每一行的poi str，使用分号分割，得到一个list
        # 然后，对于每一个list中的元素，使用空格连接起来，得到一个字符串
        # 最后，将所有的字符串连接起来，得到一个大的字符串

        # 注意，这里poi内部的元素中包含''以及'未知'，需要去除
        corpus = [' '.join(poi.split(';')) for poi in stayregions[poi_label]]
        # print(corpus)
        # 使用tfidf对每一个stay region进行向量化
        custom_stop_words = ['life','未知']
        vectorizer = TfidfVectorizer(stop_words = custom_stop_words,token_pattern='(?u)\\b\\w+\\b')
        sparse_matrix = vectorizer.fit_transform(corpus)
        # 输出词袋模型中的所有词
        word = vectorizer.get_feature_names_out()
        print(word)
        # sparse_matrix is a csr_matrix
        # print(type(sparse_matrix))
        
        print(sparse_matrix.shape)

        return sparse_matrix

    # cluster stay regions by poi tfidf
    def cluster_stayregions(self,stayregions,poi_label,cluster_label):
        # get poi tfidf
        poi_tfidf = self.get_poi_tfidf(stayregions,poi_label)
        # poi_tfidf is a csr_matrix
        # print(poi_tfidf)
        # cluster stay regions by poi tfidf value using kmeans
        kmeans = KMeans(n_clusters=35, random_state=0).fit(poi_tfidf)
        stayregions[cluster_label] = kmeans.labels_

        # print cluster labels
        # print(stayregions[cluster_label].unique())

        # print cluster label and its count
        # print(stayregions[cluster_label].value_counts()) 比较均匀

        # visualize cluster result, using the matrix of tfidf
        self.visualize_cluster_result(poi_tfidf.toarray(),stayregions[cluster_label].tolist())
        
        
        return stayregions
    
    def visualize_cluster_result(self,vectors,labels):
        # # 假设 vectors 是一个向量列表, labels 是一个标签列表
        # # 定义一个TSNE模型
        # tsne_model = TSNE(n_components=2, perplexity=40, learning_rate=100, n_iter=500)

        # # 将高维向量映射到二维空间
        # tsne_vectors = tsne_model.fit_transform(vectors)

        # # 使用matplotlib进行可视化
        # unique_labels = list(set(labels))
        # # colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(26)]
        # # 使用 seaborn 生成颜色
        # colors = sns.color_palette("husl", len(unique_labels))
        # for i, label in enumerate(unique_labels):
        #     x = tsne_vectors[np.where(np.array(labels) == label), 0]
        #     y = tsne_vectors[np.where(np.array(labels) == label), 1]
        #     plt.scatter(x, y, label=str(label),color=colors[i],s=1)
        
        # plt.legend()
        # plt.show()

        # 将向量降维到三维空间
        tsne_model = TSNE(n_components=3, perplexity=40, learning_rate=100, n_iter=500)
        tsne_vectors = tsne_model.fit_transform(vectors)

        # 使用matplotlib进行可视化
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
        for user_id,group in stayregions.groupby('user'):
            week_squences = []
            for week,week_group in group.groupby('week'):
                week_squences.append(week_group[label].tolist())
            by_week[user_id] = week_squences
        return by_week
    
    # get stay region sequence by weekday
    def by_weekday(self,stayregions,label):
        by_weekday = {}
        for user_id,group in stayregions.groupby('user'):
            weekday_squences = []
            for weekday,weekday_group in group.groupby('weekday'):
                weekday_squences.append(weekday_group[label].tolist())
            by_weekday[user_id] = weekday_squences
        return by_weekday
    
    # get stay region sequence by week with daily subset
    def by_week_with_subset(self,stayregions,label):
        by_week_with_subset = {}
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
        self.by_week = self.by_week(stayregions,label)
        self.by_weekday = self.by_weekday(stayregions,label)
        self.by_week_with_subset = self.by_week_with_subset(stayregions,label)
        self.by_weekday_with_subset = self.by_weekday_with_subset(stayregions,label)

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

        return max_frequent_patterns
   

if __name__ == '__main__':
    # first get stay regions
    stayregions = pd.read_csv('data/combined_poi.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]
    print(stayregions.shape)

    # get stay region sequence
    semc_similarity = SemcSimilarity(stayregions)
    # print result
    print(semc_similarity.stayregions)
    # cluster stay regions by poi tfidf
    stayregions = semc_similarity.cluster_stayregions(stayregions,'category1','cluster_id')
    # get stay region sequence
    semc_similarity.get_stayregion_sequence(stayregions)
    # # print result
    # print(semc_similarity.by_week)
    # print(semc_similarity.by_weekday)
    # print(semc_similarity.by_week_with_subset)
    # print(semc_similarity.by_weekday_with_subset)

    # extract prefixspan
    max_frequent_pattern_dict= semc_similarity.extract_prefixspan(semc_similarity.by_week)

    with open("./data/max_frequent_patterns_dict.pkl", "wb") as f:
        pickle.dump(max_frequent_pattern_dict, f)

    # print result
    with open("max_frequent_patterns_dict.pkl", "rb") as f:
        loaded_max_frequent_patterns_dict = pickle.load(f)

    print(loaded_max_frequent_patterns_dict)

