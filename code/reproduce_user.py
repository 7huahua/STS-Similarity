# 这个文件用于重新处理用户数据，包括删选4天2周的用户、去除pattern过少的用户、对于pattern过多的用户下采样、过少的上采样
# 首先，导入需要的库
import pandas as pd
import numpy as np
import os
from get_semc_sequence import SemcSequence
from sklearn.feature_extraction.text import TfidfVectorizer

class GetUser(object):
    # 初始化
    def __init__(self, stayregions = None,sample_num = None, sample_weight = None):
        semc_sequence = SemcSequence(stayregions)
        self.stayregions = semc_sequence.stayregions
        # self.filter_user(self.stayregions)
        self.stayregions = self.reconstruct_user(self.stayregions)
        self.resample_user(self.stayregions, sample_num, sample_weight)

    # 检查用户pattern数量是否符合要求
    def check_pattern_num(df, min_pattern_num, max_pattern_num):
        pass

    # 处理用户和它的ground truth分布不一样的情况
    def process_user_gt(self, stayregions):
        # 1. 将一组用户的多样性对比
        # 2.将数量对比
        pass

    # 筛选用户
    def filter_user(self, stayregions):
        # 对于stayregion根据用户id进行分组
        grouped = stayregions.groupby('user')
        # 对于每一个用户，检查其是否有至少2周、每周有4天的数据
        # 如果有，则保留，否则删除
        for user, group in grouped:
            # 输出用户的id和周数和天数，cluster_id的数量以及一共有多少个点
            

            print('user: {}, week: {}, day: {}, num: {}, clusters:{}'.format(user, len(group['week'].unique()), len(group['day'].unique()), len(group),len(group['cluster_id'].unique())))
            # print('user: {}, week: {}'.format(user, len(group['week'].unique())))
            # 检查用户是否有至少2周的数据
            if len(group['week'].unique()) < 2:
                stayregions.drop(group.index, inplace=True)
                continue
            # 检查用户是否有每周至少4天的数据
            for week in group['week'].unique():
                if len(group[group['week'] == week]) < 4:
                    stayregions.drop(group.index, inplace=True)
                    break
            
        print('After filter, the number of users is {}'.format(len(stayregions['user'].unique())))

    def reconstruct_user(self, stayregions):
        # 对于stayregion根据用户id进行分组
        grouped = stayregions.groupby('user')
        
        # 对于每一个用户，检查其以及对应的user+200用户包含多少种cluster_id
        # 如果差距过大，则将user+200用户的cluster_id改为user的cluster_id
        for user, group in grouped:
            if user >= 200:
                break
            # 输出cluster_id的数量
            user_cluster_num = len(group['cluster_id'].unique())
            gt_cluster_num = len(stayregions[stayregions['user'] == user + 200]['cluster_id'].unique())
            print('user: {}, user_cluster_num: {}, gt_cluster_num: {}'.format(user, user_cluster_num, gt_cluster_num))
            
        #     user_cluster_list = group['cluster_id'].unique()
        #     gt_cluster_list = stayregions[stayregions['user'] == user + 200]['cluster_id'].unique()

        #     # 如果user的cluster_id数量大于gt的cluster_id数量
        #     # 则将对应多出来的最后的cluster_id改为gt的“user”
        #     if user_cluster_num > gt_cluster_num:
        #         diff = int((user_cluster_num-gt_cluster_list)/2)
        #         stayregions.loc[(stayregions['user'] == user) & (stayregions['cluster_id'].isin(user_cluster_list[:-diff])), 'user'] = user + 200

        #         # stayregions.loc[stayregions['user'] == user, 'cluster_id'==user_cluster_list[:-diff]] = stayregions.loc[stayregions['user'] == user + 200, 'cluster_id'==user_cluster_list[:-diff]]

        #     # 如果user的cluster_id数量小于gt的cluster_id数量
        #     # 则将对应多出来的前面的cluster_id改为user的“user”
        #     elif user_cluster_num < gt_cluster_num:
        #         diff = int((gt_cluster_num-user_cluster_num)/2)
        #         stayregions.loc[(stayregions['user'] == user+200) & (stayregions['cluster_id'].isin(gt_cluster_list[:diff+1])), 'user'] = user 

        #         # stayregions.loc[stayregions['user'] == user + 200, 'cluster_id'==gt_cluster_list[:diff+1]] = stayregions.loc[stayregions['user'] == user, 'cluster_id'==gt_cluster_list[diff+1]]

        #     # sort stayregions by user and time
        #     stayregions.sort_values(by=['user', 'arr_t'], inplace=True)
            
        #     # print the cluster numbers of user and user+200
        #     print('user: {}, user_cluster_num: {}, gt_cluster_num: {}'.format(user, len(stayregions[stayregions['user'] == user]['cluster_id'].unique()), len(stayregions[stayregions['user'] == user + 200]['cluster_id'].unique())))

        # return stayregions


    def resample_user(self, stayregions, sample_num, sample_weight):

        # 假设你的DataFrame名为stayregions，已经包含了'user_id'和'cluster_id'列
        # 下面的代码将基于'user_id'分组，并计算每个组的样本数目
        grouped = stayregions.groupby('user')
        sample_counts = grouped.size()

        if sample_num == 'mean':
            # 计算样本数量的平均值作为目标采样数量
            target_num_records = int(sample_counts.mean())
            print("Target number of records per user: {}".format(target_num_records))
        
        elif sample_num == 'median':
            # 计算样本数量的中位数作为目标采样数量
            target_num_records = int(sample_counts.median())
            print("Target number of records per user: {}".format(target_num_records))
        
        else:
            print("请输入正确的采样数量方式，mean或者median")

        # 计算整体 cluster_id 的权重
        global_cluster_weights = stayregions['cluster_id'].value_counts(normalize=True).to_dict()

        # 初始化一个空的DataFrame，用于存放采样后的数据
        resampled_data = pd.DataFrame()
    
        if sample_weight == 'tfidf':

            # 创建一个包含每个用户的 "句子" 的列表，其中 cluster_id 是 "单词"
            sentences = []
            user_list = []
            for user_id, group in grouped:
                sentence = " ".join(str(x) for x in group['cluster_id'].values)
                sentences.append(sentence)
                user_list.append(user_id)

            # 对于每个用户，分别计算其 TF和IDF（逆文档频率） 值
            


            # 使用 TfidfVectorizer 计算 TF-IDF 值
            vectorizer = TfidfVectorizer(token_pattern=r'\b\d+\b')

            # idf
            tfidf_matrix = vectorizer.fit_transform(sentences)

            tfidf_mat_dict = {user_list[idx]: tfidf_matrix[idx] for idx in range(len(user_list))}

            # 将 TF-IDF 值应用于采样权重
            for user_id, group in grouped:
                cluster_counts = group['cluster_id'].value_counts(normalize=True)
                

                tfidf_values = tfidf_mat_dict[user_id].toarray()[0]

                # 获取 cluster_id 和对应的 TF-IDF 值的字典
                # tfidf_weights = {int(feature_name): tfidf_values[idx] for idx, feature_name in enumerate(vectorizer.get_feature_names_out()) if tfidf_values[idx] > 0}
                tfidf_weights_dict = {int(feature_name): tfidf_values[idx] for idx, feature_name in enumerate(vectorizer.get_feature_names_out()) if tfidf_values[idx] > 0}
                tfidf_weights = group['cluster_id'].map(tfidf_weights_dict).values

                # local weight需要根据cluster_id的数量进行归一化
                local_weights = group['cluster_id'].map(cluster_counts).values

                weights = tfidf_weights + local_weights
                # 计算 cluster_id 的最终权重
                # weights = tfidf_weights * local_weights

                if np.isnan(weights).all():
                    # 使用本地权重
                    weights = local_weights
                
                replace = len(group) < target_num_records
                resampled_records = group.sample(target_num_records, replace=True, weights=weights)
                resampled_data = pd.concat([resampled_data, resampled_records], axis=0)

        elif sample_weight == 'local':

            for user_id, group in grouped:
                cluster_counts = group['cluster_id'].value_counts(normalize=True)

                # 使用local值作为采样权重进行上采样或下采样
                replace = len(group) < target_num_records
                weights = group['cluster_id'].map(cluster_counts).values
                resampled_records = group.sample(target_num_records, replace=replace, weights=weights)
                resampled_data = pd.concat([resampled_data, resampled_records], axis=0)


        # 重置索引
        resampled_data.reset_index(drop=True, inplace=True)

        print(resampled_data)
        print("Number of records in resampled data: {}".format(len(resampled_data)))
        print("Average number of records per user: {}".format(len(resampled_data) / len(resampled_data['user'].unique())))

        # save resampled data
        resampled_data.to_csv('data/user_profile/reconstruct_data_dropped_bymean_bylocal.csv',index=False)

        return resampled_data


    # 下采样
    def down_sample(src, dst):
        pass

    # 上采样
    def up_sample(src, dst):
        pass

if __name__ == "__main__":
     # first get stay regions
    stayregions = pd.read_csv('data/combined_poi.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]
    # drop user 29 and user 229
    stayregions = stayregions[stayregions['user'] != 29]
    stayregions = stayregions[stayregions['user'] != 229]


    
    # drop 18 218 19 219 23 223 28 228 51 251 56 256 74 274 103 303 111 311 113 313 169 369
    drop_list = [18,218,19,219,23,223,28,228,51,251,56,256,74,274,103,303,111,311,113,313,169,369]

    stayregions = stayregions[~stayregions['user'].isin(drop_list)]

    get_user = GetUser(stayregions,sample_num='mean',sample_weight='local')

