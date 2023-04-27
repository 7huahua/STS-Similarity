import pandas as pd
import numpy as np
from get_semc_sequence import SemcSequence
from sklearn.feature_extraction.text import TfidfVectorizer

class GetUser(object):
    # init
    def __init__(self, stayregions = None,sample_num = None, sample_weight = None):
        semc_sequence = SemcSequence(stayregions)
        self.stayregions = semc_sequence.stayregions
        self.resample_user(self.stayregions, sample_num, sample_weight)

    def resample_user(self, stayregions, sample_num, sample_weight):

        grouped = stayregions.groupby('user')
        sample_counts = grouped.size()

        if sample_num == 'mean':
            target_num_records = int(sample_counts.mean())
            print("Target number of records per user: {}".format(target_num_records))
        
        elif sample_num == 'median':
            target_num_records = int(sample_counts.median())
            print("Target number of records per user: {}".format(target_num_records))
        
        else:
            print("please input a valid sample_num, mean or median")

        # compute the global cluster weights
        global_cluster_weights = stayregions['cluster_id'].value_counts(normalize=True).to_dict()

        # initialize the resampled data
        resampled_data = pd.DataFrame()
    
        if sample_weight == 'tfidf':

            sentences = []
            user_list = []
            for user_id, group in grouped:
                sentence = " ".join(str(x) for x in group['cluster_id'].values)
                sentences.append(sentence)
                user_list.append(user_id)

            # use TfidfVectorizer to compute TF-IDF value for every cluster_id
            vectorizer = TfidfVectorizer(token_pattern=r'\b\d+\b')

            # idf
            tfidf_matrix = vectorizer.fit_transform(sentences)

            tfidf_mat_dict = {user_list[idx]: tfidf_matrix[idx] for idx in range(len(user_list))}

            # use tfidf weight to resample
            for user_id, group in grouped:
                cluster_counts = group['cluster_id'].value_counts(normalize=True)
                

                tfidf_values = tfidf_mat_dict[user_id].toarray()[0]

                # get cluster_id and its TF-IDF value dict
                # tfidf_weights = {int(feature_name): tfidf_values[idx] for idx, feature_name in enumerate(vectorizer.get_feature_names_out()) if tfidf_values[idx] > 0}
                tfidf_weights_dict = {int(feature_name): tfidf_values[idx] for idx, feature_name in enumerate(vectorizer.get_feature_names_out()) if tfidf_values[idx] > 0}
                tfidf_weights = group['cluster_id'].map(tfidf_weights_dict).values

                # get local weight for balance
                local_weights = group['cluster_id'].map(cluster_counts).values

                
                # if u use tfidf weight, please try if it can be used alone, or it should be combined with other weight
                # weights = tfidf_weights * local_weights
                # weights = tfidf_weights + local_weights
                weights = tfidf_weights

                if np.isnan(weights).all():
                    # weights = local_weights
                    pass
                
                replace = len(group) < target_num_records
                resampled_records = group.sample(target_num_records, replace=True, weights=weights)
                resampled_data = pd.concat([resampled_data, resampled_records], axis=0)

        elif sample_weight == 'local':

            for user_id, group in grouped:
                cluster_counts = group['cluster_id'].value_counts(normalize=True)

                # use local weight to resample
                replace = len(group) < target_num_records
                weights = group['cluster_id'].map(cluster_counts).values
                resampled_records = group.sample(target_num_records, replace=replace, weights=weights)
                resampled_data = pd.concat([resampled_data, resampled_records], axis=0)


        # reset index
        resampled_data.reset_index(drop=True, inplace=True)

        print(resampled_data)
        print("Number of records in resampled data: {}".format(len(resampled_data)))
        print("Average number of records per user: {}".format(len(resampled_data) / len(resampled_data['user'].unique())))

        # save resampled data
        resampled_data.to_csv('data/user_profile/selected2_bymean_bylocal.csv',index=False)

        return resampled_data


if __name__ == "__main__":
     # first get stay regions
    stayregions = pd.read_csv('data/user_profile/stayregions_sample.csv',parse_dates=['arr_t','lea_t'])
    # only remain rows with value of category1 or category2
    stayregions = stayregions[stayregions['category1'].notnull() | stayregions['category2'].notnull()]

    get_user = GetUser(stayregions,sample_num='mean',sample_weight='local')

